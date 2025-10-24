# Stage 8 MVP: Designer-Led Template Approach

**Purpose**: Generate professional PDF reports using designer templates + manual data population instead of full automation

**Status**: ✅ **PROPOSED ALTERNATIVE** - Faster, lower-risk path to production

**Parent Document**: Stage8Planning.md (original 57.5-day automated MVP)

**Trade-off**: Exchange development time (57.5 days → 16.75 days) for manual labor (~25 hours during onboarding)

---

## MVP Deliverables

### What You BUILD (Technical Work - 6.5 days)

**Phase 1: Template Structure Creation** (MUST happen BEFORE designer work - 2 days):
1. ✅ **Hashtag → Client** template structure (COMPLETE - from MLCreativeReports.md)
2. ✅ **Hashtag → Creator** template structure (COMPLETE - from Stage8Planning.md section 1.1)
3. **Handle/Single Competitor → Client** template structure (benchmarking sections, comparison layout)
4. **Handle/Multiple Competitor → Client** template structure (side-by-side comparison structure)

**Phase 2: Data Extraction Scripts** (can happen in parallel with designer - 3 days):
1. `extract_creator_data.py` - Stage 7 JSON → 9 formatted creator reports (Google Sheets)
2. `extract_client_data.py` - Stages 1,6,7 → 1 client executive dashboard (Google Sheets)
3. `extract_competitor_data.py` - Competitor analysis → benchmarking data (Google Sheets)

**Output Format**: ✅ **Google Sheets** (easiest to review/edit before populating templates)

### What DESIGNER BUILDS (Creative Work - 11 days)

**4 Static PDF Templates**:
1. **Template A**: Content Creator Report (2-page, mobile-optimized)
2. **Template B**: Client Executive Report (3-page, intelligence dashboard)
3. **Template C**: Single Competitor Report (3-page, benchmarking)
4. **Template D**: Comparison Competitor Report (4-page, side-by-side)

**Branding Package**:
- Visual identity system (colors, fonts, spacing, grids)
- Chart templates (bar charts, star ratings, timeline graphics)
- Icon library + brand assets (logos, dividers, backgrounds)

**Editable Format**:
- Adobe InDesign files with labeled text boxes
- OR Canva Pro templates with text fields
- OR Figma templates with text layers

---

## Workflow Per Report Type

### Hashtag Analysis → Content Creators (9 PDFs)

**Frequency**: Onboarding (~5 times), then rarely needed

| Step | Who | Time | Details |
|------|-----|------|---------|
| 1. Run pipeline Stages 1-7 | Automated | Auto | Existing ML pipeline |
| 2. Extract data + QR codes | Script | 30 sec | `python extract_creator_data.py --hashtag nutrition` → Google Sheet (with real engagement metrics) + 18 QR PNGs |
| 3. Review data | You | 15 min | Open Google Sheet, verify accuracy, edit if needed |
| 4. Populate Template A (x9) | You | ~3 hrs | Copy-paste from Sheet + insert 2 QR code images per report (~20 min each) |
| 5. Export PDFs | You | 5 min | Save as PDF from InDesign/Canva |

**Total Manual Time per Hashtag**: ~3.5 hours (for 9 creator PDFs, includes QR code insertion)

**Onboarding Total**: ~17.5 hours across 5 hashtags

---

### Hashtag Analysis → Client Executive (1 PDF)

**Frequency**: Onboarding (~5 times), biweekly ongoing

| Step | Who | Time | Details |
|------|-----|------|---------|
| 1. Run pipeline Stages 1-7 | Automated | Auto | Existing ML pipeline |
| 2. Extract data | Script | 30 sec | `python extract_client_data.py --hashtag nutrition` → Google Sheet (with real engagement metrics) |
| 3. Review data | You | 10 min | Open Google Sheet, verify bucket distributions, view counts, formulas |
| 4. Populate Template B | You | 20 min | Copy-paste from Sheet into 3-page template |
| 5. Export PDF | You | 2 min | Save as PDF |

**Total Manual Time per Hashtag**: ~30 min (for 1 client PDF)

**Onboarding Total**: ~2.5 hours across 5 hashtags

**Ongoing**: ~30 min every 2 weeks

---

### Competitor Analysis → Client (Single or Comparison)

**Frequency**:
- Single: Onboarding (~5 times)
- Comparison: Onboarding (~2 times)

**Single Competitor Report**:

| Step | Who | Time | Details |
|------|-----|------|---------|
| 1. Run pipeline Stages 1-7 for competitor | Automated | Auto | Existing ML pipeline |
| 2. Extract competitor data | Script | 30 sec | `python extract_competitor_data.py --competitor @rival_brand` → Google Sheet (with real engagement metrics) |
| 3. Review benchmarks | You | 10 min | Open Google Sheet, verify competitor vs client gaps |
| 4. Populate Template C | You | 25 min | Copy-paste from Sheet into Template C |
| 5. Export PDF | You | 2 min | Save as PDF |

**Total Manual Time per Competitor**: ~40 min

**Onboarding Total**: ~3.5 hours (5 single competitor reports)

---

**Comparison Report (Multi-Competitor)**:

Same workflow as single, but populate Template D with side-by-side data.

**Total Manual Time per Comparison**: ~60 min (more data to organize)

**Onboarding Total**: ~2 hours (2 comparison reports)

---

## Onboarding Phase Manual Work

**Assuming 5 hashtag analyses + 5 single competitors + 2 comparison reports:**

| Report Type | Quantity | Time Each | Total Time |
|-------------|----------|-----------|------------|
| Creator PDFs (9 per hashtag) | 5 hashtags | 3.5 hrs | 17.5 hrs |
| Client PDFs (1 per hashtag) | 5 hashtags | 30 min | 2.5 hrs |
| Single Competitor PDFs | 5 reports | 40 min | 3.5 hrs |
| Comparison PDFs | 2 reports | 60 min | 2 hrs |
| **TOTAL ONBOARDING** | - | - | **25.5 hrs** |

**Ongoing (Post-Onboarding)**:
- Biweekly client reports: ~30 min every 2 weeks (~2 hours/month)

---

## MVP Task Breakdown

### Section 0: Template Structure Creation (2 days) - **MUST COMPLETE BEFORE DESIGNER STARTS**

| # | Task | Owner | Effort | Status | Notes |
|---|------|-------|--------|--------|-------|
| 0.1 | Hashtag → Client structure | You | 0 days | ✅ **COMPLETE** | From MLCreativeReports.md |
| 0.2 | Hashtag → Creator structure | You | 0 days | ✅ **COMPLETE** | Stage8MVP_Reports.md Section 2 (all 6 issues resolved) |
| 0.3 | Handle/Single Competitor → Client | You | 0.75 days | ✅ **COMPLETE** | Stage8MVP_Reports.md Section 3 (4-page structure, dynamic fields). **REQUIRES 2 PIPELINE RUNS**: (1) `--analysis-mode recent --period 90` for posting frequency/hashtags, (2) `--analysis-mode top` for creative patterns/QR codes. Total: 4 runs per report (competitor + client baseline). |
| 0.4 | Handle/Multiple Competitor → Client | You | 0.5 days | ⏸️ **TODO** | Side-by-side comparison structure |

**Deliverables**: 4 content structure documents (similar to MLCreativeReports.md) defining:
- Page count and purpose
- Section titles and content requirements
- Data fields needed per section
- Chart/visual requirements
- Mobile optimization specs (for Template A)

**Critical Path**: Designer cannot start until all 4 structures are complete

---

### Section 0.5: Data Processing Functions (Reference Documentation)

**Purpose**: Document reusable data processing functions used by Section 3 extraction scripts

**Note**: This section provides reference documentation for function logic. Actual Python implementations live in Section 3 scripts (`extract_creator_data.py`, `extract_client_data.py`, `extract_competitor_data.py`).

---

#### 0.5.1: Content Analysis Aggregation

**Function**: `aggregate_content_classifications(bucket_path, performance_group=None)`

**Purpose**: Aggregate 120 individual Stage 2.7 classifications into frequency distributions for report generation

**Prerequisites**:
- Stage 2.7 must add `performance_group` field to each classification output
- Field values: `"top"` or `"bottom"` (based on selection_manifest.json)
- See ContentAnalysisCHILDpt2.md Decision 1 for implementation details
- This field enables filtering by performance group without manifest cross-reference

**When to Use**:
- Report 1 (Hashtag → Client): Aggregate across all buckets for market-level content insights
- Report 2 (Hashtag → Creator): Aggregate per bucket for formula-specific content patterns
- Report 3 (Competitor): Aggregate competitor's content strategy patterns

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
4. For each of 12 key fields, calculate frequency distributions:
   - **Core Content Fields** (6): `content_category`, `hook_strategy`, `pain_points`, `keywords`, `engagement_drivers`, `content_tactics`
   - **Caption Strategy Fields** (6): `cta_type`, `emoji_usage`, `caption_length`, `hashtag_count`, `hashtag_strategy` (broad/niche/branded counts), `transcript_available`
5. Calculate effect sizes (if both top and bottom groups aggregated)

**Note on `confidence` field**: Used for filtering (Step 3), NOT included in aggregated output. This ensures only reliable classifications inform reports.

**Field Selection Rationale**: The 12 aggregated fields were chosen based on ContentAnalysisCHILDpt2.md Decision 2 (80/20 rule - highest value fields for actionable insights). Excluded fields: `caption_hook_type` (redundant with `hook_strategy`), `caption_cta_present` (90%+ have CTAs), `brand_mention_present`/`influencer_tag_present` (niche-specific), `hashtag_placement` (low variance).

**Example Implementation**:
```python
from collections import Counter
import glob
import json

def aggregate_content_classifications(bucket_path, performance_group=None):
    """
    Aggregate Stage 2.7 Content Analysis classifications.

    Returns dict with frequency distributions for 13 key fields.
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
    aggregated['caption_cta_type'] = Counter([
        c['caption_analysis']['cta_type'] for c in classifications
    ])
    aggregated['caption_emoji_usage'] = Counter([
        c['caption_analysis']['emoji_usage'] for c in classifications
    ])
    aggregated['caption_length'] = Counter([
        c['caption_analysis']['caption_length'] for c in classifications
    ])

    # Aggregate numeric fields (hashtag_count)
    hashtag_counts = [c['caption_analysis']['hashtag_count'] for c in classifications]
    aggregated['hashtag_count_stats'] = {
        'mean': sum(hashtag_counts) / len(hashtag_counts),
        'min': min(hashtag_counts),
        'max': max(hashtag_counts),
        'median': sorted(hashtag_counts)[len(hashtag_counts) // 2]
    }

    # Aggregate hashtag strategy (broad/niche/branded counts)
    broad_counts = []
    niche_counts = []
    branded_counts = []
    for c in classifications:
        hs = c['caption_analysis'].get('hashtag_strategy', {})
        broad_counts.append(hs.get('broad_count', 0))
        niche_counts.append(hs.get('niche_count', 0))
        branded_counts.append(hs.get('branded_count', 0))

    aggregated['hashtag_strategy_avg'] = {
        'avg_broad': sum(broad_counts) / len(broad_counts) if broad_counts else 0,
        'avg_niche': sum(niche_counts) / len(niche_counts) if niche_counts else 0,
        'avg_branded': sum(branded_counts) / len(branded_counts) if branded_counts else 0
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
    'caption_cta_type': Counter({
        'link_in_bio': 32,
        'save_post': 5,
        'comment': 3
    }),
    'caption_emoji_usage': Counter({
        'some': 28,
        'many': 8,
        'none': 4
    }),
    'caption_length': Counter({
        'short': 26,
        'long': 14
    }),
    'hashtag_count_stats': {
        'mean': 7.2,
        'min': 3,
        'max': 12,
        'median': 7
    },
    'hashtag_strategy_avg': {
        'avg_broad': 2.1,
        'avg_niche': 4.8,
        'avg_branded': 0.3
    },
    'transcript_available_ratio': 0.95
}

# Note: 'confidence' field is NOT in output - it was used for filtering only

```

**Usage in Reports**:
- **Report 1**: Show top content categories, hook strategies at market level
- **Report 2**: Show contrastive analysis (top vs bottom behaviors)
- **Report 3**: Show competitor's content strategy patterns

**Effect Size Calculation** (for contrastive analysis):
```python
def calculate_effect_sizes(top_stats, bottom_stats):
    """
    Calculate effect sizes for contrastive analysis.

    Effect size = top_frequency / bottom_frequency
    Example: If 60% top use problem_solution vs 20% bottom → 3.0x effect
    """
    effect_sizes = {}

    # For each field that appears in both
    for field in ['hook_strategy', 'content_category', 'engagement_drivers']:
        if field not in top_stats or field not in bottom_stats:
            continue

        top_counter = top_stats[field]
        bottom_counter = bottom_stats[field]

        # Calculate percentages and ratios
        for item in set(list(top_counter.keys()) + list(bottom_counter.keys())):
            top_pct = (top_counter.get(item, 0) / top_stats['total_videos']) * 100
            bottom_pct = (bottom_counter.get(item, 0) / bottom_stats['total_videos']) * 100

            if bottom_pct > 0:  # Avoid division by zero
                ratio = top_pct / bottom_pct

                # Only include if meaningful difference (>1.5x or <0.67x)
                if ratio > 1.5 or ratio < 0.67:
                    effect_sizes[f"{field}.{item}"] = {
                        'top_percentage': round(top_pct, 1),
                        'bottom_percentage': round(bottom_pct, 1),
                        'effect_size': round(ratio, 1)
                    }

    return effect_sizes

# Example output:
{
    'hook_strategy.problem_solution': {
        'top_percentage': 60.0,
        'bottom_percentage': 20.0,
        'effect_size': 3.0
    },
    'engagement_drivers.before_after_reveal': {
        'top_percentage': 47.5,
        'bottom_percentage': 15.0,
        'effect_size': 3.2
    }
}
```

---

#### 0.5.1.1: Top N Selection Wrapper

**Function**: `get_top_n_from_field(bucket_path, field_name, n=3, performance_group="top")`

**Purpose**: Wrapper function to extract Top N items from a specific classification field for report display

**Why This Exists**:
- Base function `aggregate_content_classifications()` returns ALL data as Counter objects
- Reports need specific formatted output: Array of top N strings
- This wrapper adapts the base function output to report field requirements

**Input Parameters**:
- `bucket_path` (string): Path to bucket folder
  - Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/`
- `field_name` (string): Classification field to extract
  - Valid values: `"content_category"`, `"hook_strategy"`, `"pain_points"`, `"keywords"`, `"engagement_drivers"`, `"content_tactics"`
- `n` (int): Number of top items to return (default: 3)
  - Example: 3 for Top 3, 5 for Top 5, 8 for Top 8
- `performance_group` (string, optional): Filter by "top" or "bottom" performers (default: "top")

**Returns**:
- Array of strings: Top N item names ranked by frequency
- Example: `["recipe_tutorial", "wellness_practice", "supplement_review"]`

**Example Implementation**:
```python
def get_top_n_from_field(bucket_path, field_name, n=3, performance_group="top"):
    """
    Extract Top N items from a classification field.

    Wrapper around aggregate_content_classifications() that:
    1. Calls base aggregation function
    2. Extracts specific field
    3. Returns Top N items as array of strings
    """
    # Call base aggregation function
    result = aggregate_content_classifications(bucket_path, performance_group)

    if result is None:
        return []  # No data found

    # Validate field name
    if field_name not in result:
        raise ValueError(f"Invalid field_name: {field_name}. Must be one of: {list(result.keys())}")

    # Extract Counter object for this field
    counter = result[field_name]

    # Get Top N items (returns list of tuples: [(item, count), ...])
    top_n_tuples = counter.most_common(n)

    # Extract just the item names (ignore counts)
    top_n_items = [item for item, count in top_n_tuples]

    return top_n_items
```

**Usage Examples**:

```python
# Report 2: VIDEO CATEGORY Section - Top 3 Content Categories
top_3_categories = get_top_n_from_field(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/",
    field_name="content_category",
    n=3,
    performance_group="top"
)
# Returns: ["recipe_tutorial", "wellness_practice", "supplement_review"]

# Report 2: VIDEO CATEGORY Section - Top 3 Engagement Drivers
top_3_drivers = get_top_n_from_field(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/",
    field_name="engagement_drivers",
    n=3,
    performance_group="top"
)
# Returns: ["personal_testimony", "before_after_reveal", "product_demonstration"]

# Report 2: HOOK Section - Top 3 Hook Strategies
top_3_hooks = get_top_n_from_field(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/",
    field_name="hook_strategy",
    n=3,
    performance_group="top"
)
# Returns: ["question_hook", "problem_solution", "shocking_fact"]

# Report 2: BUILD & PROVE Section - Top 8 Keywords
top_8_keywords = get_top_n_from_field(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/",
    field_name="keywords",
    n=8,
    performance_group="top"
)
# Returns: ["protein", "gut_health", "fiber", "probiotics", "metabolism", "holistic", "meal_prep", "supplements"]

# Report 2: BUILD & PROVE Section - Top 5 Pain Points
top_5_pain_points = get_top_n_from_field(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/",
    field_name="pain_points",
    n=5,
    performance_group="top"
)
# Returns: ["bloating", "low_energy", "weight_loss", "gut_health", "brain_fog"]

# Report 2: BUILD & PROVE Section - Top 4 Content Tactics
top_4_tactics = get_top_n_from_field(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/",
    field_name="content_tactics",
    n=4,
    performance_group="top"
)
# Returns: ["direct_to_camera", "voiceover", "text_overlay", "product_showcase"]
```

**Error Handling**:
```python
# Invalid field name
get_top_n_from_field(bucket_path, "invalid_field", n=3)
# Raises: ValueError("Invalid field_name: invalid_field. Must be one of: [...]")

# No data found (empty bucket or no classifications)
get_top_n_from_field("/path/to/empty/bucket", "keywords", n=3)
# Returns: []
```

**Relationship to Base Function**:
- **Base**: `aggregate_content_classifications()` → Returns full dataset (all fields, all counts)
- **Wrapper**: `get_top_n_from_field()` → Returns specific field, top N only, formatted for reports

---

#### 0.5.2: QR Code Video Selection (Simple Performance-Based)

**Function**: `select_qr_code_videos(bucket_path, bucket_name)`

**Purpose**: Select top and bottom performer videos for QR codes based on performance ranking (no cluster/pattern logic)

**When to Use**:
- Report 2 (Hashtag → Creator): Generate 2 QR codes per bucket (top + bottom performer examples)
- MVP approach that works without Stage 6 K-means or Stage 7 pattern identification

**Input Parameters**:
- `bucket_path` (string): Path to bucket folder
  - Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/`
- `bucket_name` (string): Duration bucket name
  - Example: `"18-33s"`, `"60-90s"`, `"13-18s"`
  - Must match a bucket in `selection_manifest.json`

**Process**:
1. Load `selection_manifest.json` to get top/bottom performer video IDs for this bucket
2. Load `selected_videos.json` for this bucket to get video metadata
3. Filter videos by `top_performers` and `bottom_performers` arrays
4. Select top video: Highest views from top_performers (newest timestamp as tiebreaker)
5. Select bottom video: Highest views from bottom_performers (newest timestamp as tiebreaker)
6. Return URLs and metadata for QR code generation

**Example Implementation**:
```python
import json

def select_qr_code_videos(bucket_path, bucket_name):
    """
    Select top and bottom performer videos for QR codes (simple performance-based).

    Returns dict with top/bottom video URLs and metadata.

    No cluster logic - just selects from top_performers vs bottom_performers arrays.
    Videos are dynamically matched to the bucket duration.
    """
    # Step 1: Load selection manifest
    manifest_path = f"{bucket_path}/../../selection_manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Get video IDs for THIS bucket from manifest
    bucket_data = manifest['videos_by_bucket'][bucket_name]
    top_ids = bucket_data['top_performers']
    bottom_ids = bucket_data['bottom_performers']

    # Step 2: Load selected videos for this bucket
    selected_path = f"{bucket_path}/selected_videos.json"
    with open(selected_path, 'r') as f:
        selected_data = json.load(f)

    # Step 3: Filter videos by performance group
    top_videos = [v for v in selected_data['videos'] if v['id'] in top_ids]
    bottom_videos = [v for v in selected_data['videos'] if v['id'] in bottom_ids]

    # Step 4: Select videos (highest playCount, newest createTime as tiebreaker)
    top_video = max(top_videos, key=lambda v: (v['playCount'], v['createTime']))
    bottom_video = max(bottom_videos, key=lambda v: (v['playCount'], v['createTime']))

    return {
        'top_performer': {
            'video_id': top_video['id'],
            'url': top_video['webVideoUrl'],
            'views': top_video['playCount'],
            'timestamp': top_video['createTime'],
            'engagement_data': {
                'diggCount': top_video['diggCount'],
                'commentCount': top_video['commentCount'],
                'shareCount': top_video['shareCount'],
                'collectCount': top_video['collectCount']
            }
        },
        'bottom_performer': {
            'video_id': bottom_video['id'],
            'url': bottom_video['webVideoUrl'],
            'views': bottom_video['playCount'],
            'timestamp': bottom_video['createTime'],
            'engagement_data': {
                'diggCount': bottom_video['diggCount'],
                'commentCount': bottom_video['commentCount'],
                'shareCount': bottom_video['shareCount'],
                'collectCount': bottom_video['collectCount']
            }
        }
    }
```

**Output Format**:
```python
{
    'top_performer': {
        'video_id': '7545713916584774968',
        'url': 'https://www.tiktok.com/@agitthaaii/video/7545713916584774968',
        'views': 11100000,
        'timestamp': 1756873437,
        'engagement_data': {
            'diggCount': 133200,
            'commentCount': 2664,
            'shareCount': 26640,
            'collectCount': 11100
        }
    },
    'bottom_performer': {
        'video_id': '7560886598309612814',
        'url': 'https://www.tiktok.com/@ahealthydoseofash/video/7560886598309612814',
        'views': 490,
        'timestamp': 1760406106,
        'engagement_data': {
            'diggCount': 12,
            'commentCount': 0,
            'shareCount': 1,
            'collectCount': 3
        }
    }
}
```

**Usage Example**:
```python
# For Report 2 covering 18-33s bucket
bucket_path = "/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s"
bucket_name = "18-33s"

video_data = select_qr_code_videos(bucket_path, bucket_name)
# Returns URLs for videos that are 18-33 seconds long

# For Report 2 covering 60-90s bucket
bucket_path = "/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_60-90s"
bucket_name = "60-90s"

video_data = select_qr_code_videos(bucket_path, bucket_name)
# Returns URLs for videos that are 60-90 seconds long
```

**QR Code Generation** (from selected videos):
```python
import qrcode

def generate_qr_codes(video_data, output_dir, bucket_name):
    """Generate QR code PNG files for top and bottom videos."""

    # Generate top performer QR code
    qr_top = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr_top.add_data(video_data['top_performer']['url'])
    qr_top.make()
    img_top = qr_top.make_image(fill_color="black", back_color="white")
    img_top.save(f"{output_dir}/{bucket_name}_top_performer.png")

    # Generate bottom performer QR code
    qr_bottom = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr_bottom.add_data(video_data['bottom_performer']['url'])
    qr_bottom.make()
    img_bottom = qr_bottom.make_image(fill_color="black", back_color="white")
    img_bottom.save(f"{output_dir}/{bucket_name}_bottom_performer.png")

    return {
        'top_qr_path': f"{output_dir}/{bucket_name}_top_performer.png",
        'bottom_qr_path': f"{output_dir}/{bucket_name}_bottom_performer.png"
    }
```

**Validation Status**: ✅ **READY** (Data exists: selection_manifest.json + selected_videos.json)

**Key Features**:
- ✅ Dynamic bucket selection (videos match report duration)
- ✅ No Stage 6/7 dependencies (works with current data)
- ✅ Simple performance-based logic (top vs bottom)
- ✅ Verified with actual data (tested on 18-33s and 60-90s buckets)

**Future Enhancement**:
When Stage 6 K-means and Stage 7 pattern identification are implemented, this function can be extended to support cluster-based selection by adding an optional `formula_cluster_id` parameter.

---

#### 0.5.3: Hashtag Extraction

**Function**: `extract_hashtag_analysis(manifest_path)`

**Purpose**: Extract top 10 hashtags from selected videos (for competitor/handle analysis)

**When to Use**:
- Report 3 (Competitor): Show top hashtags competitor uses

**Documentation**: See Stage8MVP_Reports.md Section 3, Item #2 (lines 1352-1400) for complete implementation details.

**Quick Reference**:
- **Input**: `selection_manifest.json` path
- **Output**: Top 10 hashtags with usage percentages
- **Source Data**: `unified_analysis/{video_id}.json` → `metadata.hashtags` array

---

#### 0.5.4: @Mention Extraction

**Function**: `extract_mention_analysis(manifest_path)`

**Purpose**: Extract @mentions to identify affiliate/repost partnerships

**When to Use**:
- Report 3 (Competitor): Analyze competitor's content sourcing strategy (original vs reposted)

**Documentation**: See Stage8MVP_Reports.md Section 3, Item #3 (lines 1402-1505) for complete implementation details.

**Quick Reference**:
- **Input**: `selection_manifest.json` path
- **Output**: Top 10 @mentions, repost rate percentage
- **Source Data**: `unified_analysis/{video_id}.json` → `metadata.description` (extract via regex)

---

#### 0.5.5: Engagement Rate Calculation

**Function**: `calculate_engagement_metrics(video_metadata)`

**Purpose**: Calculate real engagement rate from Apify metadata (views, likes, comments, shares, saves)

**When to Use**:
- All reports (Templates 1-4): Show real engagement performance alongside view metrics
- Replaces "Industry Benchmark Mapping" estimation method

**Input Parameters**:
- `video_metadata` (dict): Video metadata from `unified_analysis/{video_id}.json`
  - Source location: Lines 8-12 of metadata section
  - Required fields: `views`, `likes`, `comments`, `shares`, `saves`

**Process**:
1. Load metadata from `unified_analysis/{video_id}.json` → `metadata` (lines 8-12)
2. Extract: `views`, `likes`, `comments`, `shares`, `saves`
3. Calculate total interactions: likes + comments + shares + saves
4. Calculate engagement rate: (total_interactions / views) × 100%
5. Calculate individual metric rates (likes_rate, comments_rate, shares_rate, saves_rate)

**Example Implementation**:
```python
def calculate_engagement_metrics(video_metadata):
    """
    Calculate engagement rate from Apify metadata.

    Source: unified_analysis/{video_id}.json → metadata (lines 8-12)

    Returns dict with engagement rates and interaction counts.
    """
    views = video_metadata.get('views', 0)
    if views == 0:
        return {
            'engagement_rate': 0.0,
            'total_interactions': 0,
            'likes_rate': 0.0,
            'comments_rate': 0.0,
            'shares_rate': 0.0,
            'saves_rate': 0.0
        }

    likes = video_metadata.get('likes', 0)
    comments = video_metadata.get('comments', 0)
    shares = video_metadata.get('shares', 0)
    saves = video_metadata.get('saves', 0)

    total_interactions = likes + comments + shares + saves
    engagement_rate = (total_interactions / views) * 100

    return {
        'engagement_rate': round(engagement_rate, 2),  # e.g., 1.23%
        'total_interactions': total_interactions,
        'likes_rate': round((likes / views) * 100, 2),
        'comments_rate': round((comments / views) * 100, 2),
        'shares_rate': round((shares / views) * 100, 2),
        'saves_rate': round((saves / views) * 100, 2)
    }
```

**Output Format**:
```python
{
    'engagement_rate': 1.23,  # Percentage
    'total_interactions': 2507,  # Sum of all engagement actions
    'likes_rate': 0.86,  # Percentage
    'comments_rate': 0.03,  # Percentage
    'shares_rate': 0.12,  # Percentage
    'saves_rate': 0.11  # Percentage
}
```

**Usage in Reports**:
- **Template 1 (Client)**: Show engagement alongside views in performance by duration table
- **Template 2 (Creator)**: Show real engagement comparison (top vs bottom cluster)
- **Template 3 (Competitor)**: Show engagement in bucket performance comparison
- **Template 4 (Multi-Competitor)**: Show engagement in cross-competitor performance matrix

**Data Source**:
```json
// From unified_analysis/{video_id}.json → metadata (lines 8-12)
{
  "metadata": {
    "views": 223700,
    "likes": 1923,
    "comments": 65,
    "shares": 266,
    "saves": 253
  }
}
```

**Comparison to Industry Benchmark Mapping** (Previous Method):
- **OLD**: Estimated engagement rates based on view performance tiers (6-9% for top, 2-4% for bottom)
- **NEW**: Real calculated engagement from actual TikTok interaction data
- **Benefit**: Data integrity, transparency, no estimation needed

---

#### 0.5.6: Report 4 Multi-Competitor Functions (NEW)

**Purpose**: Functions required for Report 4 Performance Rankings (Multi-Competitor Analysis). These 5 new functions aggregate competitor data across winning buckets for comparative analysis.

**Context**: Report 4 compares 3-5 competitors side-by-side. Each competitor has their own analysis directory with 3 winning buckets. These functions aggregate data per competitor for the Performance Rankings table.

---

##### Function 1: `calculate_competitor_avg_views()`

**Function**: `calculate_competitor_avg_views(client_id: str, competitor_handle: str) -> int`

**Purpose**: Calculate weighted average views for a competitor across 3 winning buckets

**Type**: 🆕 Entirely new function

**Reuses**: Data structure from `selected_videos.json` (existing file)

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitor_handle` (str): Competitor handle with @ symbol
  - Example: `"@nike"`

**Process**:
1. **Discover analysis directory dynamically**:
   - Remove `@` from handle: `"@nike"` → `"nike"`
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - List directories starting with `"top_"` (e.g., `"top_contrastive"` or `"top_top"`)
   - Use first directory found (should be only one)
2. Identify 3 winning buckets from `winner_analysis.json → top_3_buckets`
3. For each winning bucket:
   - Load `buckets/bucket_{name}/selected_videos.json`
   - Get `top_count` (e.g., 33)
   - Extract first `top_count` videos (top performers, sorted by playCount DESC)
   - Calculate bucket average: `sum(playCount) / top_count`
4. Calculate weighted average across 3 buckets:
   - Formula: `Σ(bucket_avg × top_count) / Σ(top_count)`

**Example Implementation**:
```python
def calculate_competitor_avg_views(client_id: str, competitor_handle: str) -> int:
    """
    Calculate weighted average views across winning buckets.
    Dynamically discovers the analysis directory per competitor.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitor_handle: Competitor handle with @ symbol (e.g., "@nike")

    Returns:
        int: Weighted average views (e.g., 580000)

    Example:
        >>> calculate_competitor_avg_views("test_run", "@nike")
        580000  # 580K avg views across 3 buckets
    """
    import os
    import json
    import logging

    logger = logging.getLogger(__name__)

    # Remove @ symbol for directory name
    competitor_dir = competitor_handle.lstrip('@')

    # Base path
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

    # Discover analysis directory (should be only one starting with "top_")
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(
            f"No analysis directory found for {competitor_handle} at {base_path}"
        )

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]  # e.g., ["3-9s", "9-13s", "18-33s"]

    bucket_stats = []

    for bucket in winning_buckets:
        # Load selected videos
        with open(f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json") as f:
            data = json.load(f)

        top_count = data["top_count"]
        top_videos = data["videos"][:top_count]  # First N videos = top performers

        # Calculate bucket average
        bucket_avg = sum(v["playCount"] for v in top_videos) / len(top_videos)

        bucket_stats.append({
            "bucket": bucket,
            "avg_views": bucket_avg,
            "video_count": top_count
        })

    # Weighted average
    total_views = sum(b["avg_views"] * b["video_count"] for b in bucket_stats)
    total_videos = sum(b["video_count"] for b in bucket_stats)

    weighted_avg = int(total_views / total_videos)

    return weighted_avg
```

**Output**: Integer (e.g., 580000) - format with K/M suffix in report display (580K)

**Used in**: Report 4 → Performance Rankings → Field #2 (Avg Views column)

---

##### Function 2: `calculate_posting_frequency()`

**Function**: `calculate_posting_frequency(client_id: str, competitor_handle: str) -> float`

**Purpose**: Calculate posting frequency (videos per week) from date-filtered video count

**Type**: 🆕 Entirely new function

**Reuses**: Data from `winner_analysis.json` and `config.json` (existing files)

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitor_handle` (str): Competitor handle with @ symbol
  - Example: `"@nike"`

**Process**:
1. **Discover analysis directory dynamically**:
   - Remove `@` from handle: `"@nike"` → `"nike"`
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - List directories starting with `"top_"` (e.g., `"top_contrastive"` or `"top_top"`)
   - Use first directory found (should be only one)
2. Load `winner_analysis.json → top_100_distribution`
3. Sum all bucket values: `total_videos = sum(top_100_distribution.values())`
   - This represents total videos in date range (exact if <100, minimum if =100)
4. Load `config.json → date_filter` (e.g., "last_90_days")
5. Extract days: `days = int(date_filter.replace("last_", "").replace("_days", ""))`
6. Calculate weeks: `weeks = days / 7`
7. Calculate frequency: `videos_per_week = total_videos / weeks`

**Example Implementation**:
```python
def calculate_posting_frequency(client_id: str, competitor_handle: str) -> float:
    """
    Calculate posting frequency (videos per week).
    Dynamically discovers the analysis directory per competitor.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitor_handle: Competitor handle with @ symbol (e.g., "@nike")

    Returns:
        float: Videos per week (e.g., 7.5)

    Example:
        >>> calculate_posting_frequency("test_run", "@drinkpoppi")
        7.6  # 98 videos / 12.86 weeks
    """
    import os
    import json

    # Remove @ symbol for directory name
    competitor_dir = competitor_handle.lstrip('@')

    # Base path
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

    # Discover analysis directory (should be only one starting with "top_")
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(
            f"No analysis directory found for {competitor_handle} at {base_path}"
        )

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winner analysis
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)

    # Sum videos from top_100_distribution
    total_videos = sum(winner_data["top_100_distribution"].values())
    # Example: {"3-9s": 35, "9-13s": 18, "18-33s": 15, ...} → sum = 98

    # Load config
    with open(f"{competitor_path}/config.json") as f:
        config = json.load(f)

    # Extract days from date_filter
    date_filter = config["date_filter"]  # "last_90_days"
    days = int(date_filter.replace("last_", "").replace("_days", ""))  # 90

    # Calculate weeks
    weeks = days / 7  # 90 / 7 = 12.86

    # Calculate posting frequency
    posting_freq = round(total_videos / weeks, 1)  # 98 / 12.86 = 7.6

    return posting_freq
```

**Output**: Float (e.g., 7.5) - displayed as "7.5 videos/week"

**Confidence Level**:
- If `total_videos < 100`: "exact" (degraded mode analyzed all videos)
- If `total_videos == 100`: "minimum" (might be more videos beyond top 100)

**Used in**: Report 4 → Performance Rankings → Field #3 (Posting Freq column)

**Verified Example**: Drinkpoppi = 98 videos / 13 weeks = 7.5 videos/week

---

##### Function 3: `calculate_videos_analyzed()`

**Function**: `calculate_videos_analyzed(client_id: str, competitor_handle: str) -> int`

**Purpose**: Sum total videos analyzed across winning buckets

**Type**: 🆕 Entirely new function

**Reuses**: Data structure from `selected_videos.json` (existing file)

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitor_handle` (str): Competitor handle with @ symbol
  - Example: `"@nike"`

**Process**:
1. **Discover analysis directory dynamically**:
   - Remove `@` from handle: `"@nike"` → `"nike"`
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - List directories starting with `"top_"` (e.g., `"top_contrastive"` or `"top_top"`)
   - Use first directory found (should be only one)
2. Identify 3 winning buckets from `winner_analysis.json → top_3_buckets`
3. For each winning bucket:
   - Load `buckets/bucket_{name}/selected_videos.json`
   - Extract `selected_count` (e.g., 42)
4. Sum: `total_analyzed = sum(selected_count for all buckets)`

**Example Implementation**:
```python
def calculate_videos_analyzed(client_id: str, competitor_handle: str) -> int:
    """
    Sum total videos analyzed across winning buckets.
    Dynamically discovers the analysis directory per competitor.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitor_handle: Competitor handle with @ symbol (e.g., "@nike")

    Returns:
        int: Total videos analyzed (e.g., 145)

    Example:
        >>> calculate_videos_analyzed("test_run", "@nike")
        145  # 50 + 45 + 50 across 3 buckets
    """
    import os
    import json

    # Remove @ symbol for directory name
    competitor_dir = competitor_handle.lstrip('@')

    # Base path
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

    # Discover analysis directory (should be only one starting with "top_")
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(
            f"No analysis directory found for {competitor_handle} at {base_path}"
        )

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]

    total_analyzed = 0

    for bucket in winning_buckets:
        # Load selected videos
        with open(f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json") as f:
            data = json.load(f)

        total_analyzed += data["selected_count"]

    return total_analyzed
```

**Output**: Integer (e.g., 145)

**Used in**: Report 4 → Performance Rankings → Field #4 (Videos Analyzed column)

---

##### Function 4: `calculate_competitor_avg_engagement()`

**Function**: `calculate_competitor_avg_engagement(client_id: str, competitor_handle: str) -> float`

**Purpose**: Calculate average engagement rate for competitor's top performers across winning buckets

**Type**: 🔄 NEW wrapper function (adaptation of existing base function)

**Reuses**: ✅ Calls existing `calculate_engagement_metrics()` (Section 0.5.5)

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitor_handle` (str): Competitor handle with @ symbol
  - Example: `"@nike"`

**Process**:
1. **Discover analysis directory dynamically**:
   - Remove `@` from handle: `"@nike"` → `"nike"`
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - List directories starting with `"top_"` (e.g., `"top_contrastive"` or `"top_top"`)
   - Use first directory found (should be only one)
2. Identify 3 winning buckets from `winner_analysis.json → top_3_buckets`
3. For each winning bucket:
   - Load `buckets/bucket_{name}/selected_videos.json`
   - Get top performer video IDs (first `top_count` videos)
   - For each video:
     - Load `/buckets/{bucket}/analysis/unified_analysis/{video_id}.json`
     - Extract `metadata` (lines 8-12: playCount, diggCount, commentCount, shareCount, collectCount)
     - Call existing `calculate_engagement_metrics(metadata)` ← BASE FUNCTION
     - Collect engagement_rate
4. Average all engagement rates across all top performers from 3 buckets

**Example Implementation**:
```python
def calculate_competitor_avg_engagement(client_id: str, competitor_handle: str) -> float:
    """
    Calculate average engagement rate across top performers.
    Dynamically discovers the analysis directory per competitor.

    Adaptation: Uses existing calculate_engagement_metrics() as base.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitor_handle: Competitor handle with @ symbol (e.g., "@nike")

    Returns:
        float: Average engagement % (e.g., 1.4)

    Example:
        >>> calculate_competitor_avg_engagement("test_run", "@nike")
        1.4  # 1.4% average engagement
    """
    import os
    import json

    # Remove @ symbol for directory name
    competitor_dir = competitor_handle.lstrip('@')

    # Base path
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

    # Discover analysis directory (should be only one starting with "top_")
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(
            f"No analysis directory found for {competitor_handle} at {base_path}"
        )

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]

    all_engagement_rates = []

    for bucket in winning_buckets:
        # Load selected videos
        with open(f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json") as f:
            data = json.load(f)

        top_count = data["top_count"]
        top_videos = data["videos"][:top_count]  # First N = top performers

        for video in top_videos:
            video_id = video["id"]

            # Load unified analysis metadata
            unified_path = f"{competitor_path}/buckets/bucket_{bucket}/analysis/unified_analysis/{video_id}.json"
            with open(unified_path) as f:
                unified_data = json.load(f)

            metadata = unified_data["metadata"]

            # ← CALL EXISTING BASE FUNCTION (Section 0.5.5)
            engagement_result = calculate_engagement_metrics(metadata)
            engagement_rate = engagement_result["engagement_rate"]

            all_engagement_rates.append(engagement_rate)

    # Average across all top performers
    avg_engagement = round(sum(all_engagement_rates) / len(all_engagement_rates), 2)

    return avg_engagement
```

**Output**: Float (e.g., 1.4) - displayed as "1.4%"

**Scope Limitation**: This measures engagement of **top performers only**, not all posted videos. Consistent with `calculate_competitor_avg_views()` methodology.

**Used in**: Report 4 → Performance Rankings → Field #5 (Avg Engagement column)

**Data Fields Used**: `metadata.{playCount, diggCount, commentCount, shareCount, collectCount}`

---

##### Function 5: `calculate_market_leader()`

**Function**: `calculate_market_leader(competitors: list[dict]) -> str`

**Purpose**: Determine market leader using composite score (normalized views + engagement)

**Type**: 🆕 Entirely new function

**Reuses**: Logic pattern from Report 1 "← BEST" bucket selection

**Input Parameters**:
- `competitors` (list[dict]): List of competitor data dictionaries
  - Each dict contains: `{"handle": str, "avg_views": int, "avg_engagement": float}`
  - Example: `[{"handle": "@nike", "avg_views": 580000, "avg_engagement": 1.4}, ...]`

**Process**:
1. Find max avg_views across all competitors for normalization
2. For each competitor:
   - Normalize views to 0-100 scale: `views_score = (avg_views / max_views) × 100`
   - Engagement already in % (0-100): `engagement_score = avg_engagement`
   - Calculate composite: `composite_score = views_score + engagement_score`
3. Return competitor with highest composite_score

**Example Implementation**:
```python
def calculate_market_leader(competitors: list[dict]) -> str:
    """
    Determine market leader using composite score.

    Logic: Same as Report 1 "← BEST" bucket (normalized views + engagement).

    Args:
        competitors: List of competitor data dicts
            Example: [{"handle": "@nike", "avg_views": 580000, "avg_engagement": 1.4}, ...]

    Returns:
        str: Market leader handle (e.g., "@nike")

    Example:
        >>> competitors = [
        ...     {"handle": "@nike", "avg_views": 580000, "avg_engagement": 1.4},
        ...     {"handle": "@adidas", "avg_views": 520000, "avg_engagement": 1.3},
        ...     {"handle": "@puma", "avg_views": 480000, "avg_engagement": 1.2}
        ... ]
        >>> calculate_market_leader(competitors)
        "@nike"  # Highest composite score: 101.4
    """
    best_competitor = None
    best_score = -1

    # Find max views for normalization
    max_views = max(c["avg_views"] for c in competitors)

    for competitor in competitors:
        # Normalize views to 0-100 scale
        views_score = (competitor["avg_views"] / max_views) * 100

        # Engagement already in % (0-100)
        engagement_score = competitor["avg_engagement"]

        # Composite score (engagement + views)
        composite_score = views_score + engagement_score

        if composite_score > best_score:
            best_score = composite_score
            best_competitor = competitor["handle"]

    return best_competitor
```

**Output**: String (e.g., "@nike")

**Example Calculation**:
```
Competitors:
- @nike:   (580K / 580K × 100) + 1.4% = 100.0 + 1.4 = 101.4 ← Market Leader
- @adidas: (520K / 580K × 100) + 1.3% =  89.7 + 1.3 =  91.0
- @puma:   (480K / 580K × 100) + 1.2% =  82.8 + 1.2 =  84.0
```

**Used in**: Report 4 → Performance Rankings → Field #6 (Market Leader determination)

**Logic Source**: Adapted from Report 1 → Page 2 → Section 2 → "Top bucket label" (← BEST)

---

##### Bucket Distribution Functions (Functions 6-9)

**Purpose**: Functions for Report 4 Page 2 Section 1 (Bucket Distribution Comparison). These 4 functions analyze content distribution across duration buckets to show where competitors focus content and identify market patterns.

**Context**: Shows percentage allocation across 8 duration buckets (0-3s, 3-9s, 9-13s, 13-18s, 18-33s, 33-60s, 60-90s, 90-120s) for each competitor, with market-level pattern detection.

---

##### Function 6: `calculate_bucket_distribution()`

**Function**: `calculate_bucket_distribution(winner_analysis_path: str) -> dict`

**Purpose**: Calculate percentage distribution across 8 duration buckets for a single competitor

**Type**: 🆕 Entirely new function

**Reuses**: Data from `winner_analysis.json` (existing file)

**Input Parameters**:
- `winner_analysis_path` (str): Path to winner_analysis.json
  - Example: `/data/clients/{client}/competitors/nike/top_contrastive/winner_analysis.json`

**Process**:
1. Load `winner_analysis.json → top_100_distribution`
2. Sum all bucket values to get total: `total = sum(top_100_distribution.values())`
3. For each of 8 buckets:
   - Get count from top_100_distribution (0 if bucket not present)
   - Calculate percentage: `(count / total) × 100`
   - Round to nearest integer
4. Return dict with 8 bucket percentages

**Example Implementation**:
```python
def calculate_bucket_distribution(winner_analysis_path: str) -> dict:
    """
    Calculate percentage distribution across 8 duration buckets.

    Args:
        winner_analysis_path: Path to winner_analysis.json

    Returns:
        dict: {bucket_name: percentage} for all 8 buckets

    Example:
        >>> calculate_bucket_distribution(".../drinkpoppi/.../winner_analysis.json")
        {
            "0-3s": 8,
            "3-9s": 36,
            "9-13s": 18,
            "13-18s": 5,
            "18-33s": 15,
            "33-60s": 11,
            "60-90s": 6,
            "90-120s": 0
        }
    """
    import json

    with open(winner_analysis_path) as f:
        data = json.load(f)

    top_100_dist = data["top_100_distribution"]
    total = sum(top_100_dist.values())  # e.g., 98

    # Calculate percentages for all 8 buckets
    all_buckets = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
    bucket_percentages = {}

    for bucket in all_buckets:
        count = top_100_dist.get(bucket, 0)  # 0 if bucket not present
        percentage = round(count / total * 100, 0) if total > 0 else 0
        bucket_percentages[bucket] = int(percentage)

    return bucket_percentages
```

**Output**: Dict with 8 integer percentages (e.g., `{"0-3s": 8, "3-9s": 36, ...}`)

**Verified Example**: Drinkpoppi = `{"0-3s": 8, "3-9s": 36, "9-13s": 18, "13-18s": 5, "18-33s": 15, "33-60s": 11, "60-90s": 6, "90-120s": 0}` (sum = 99% due to rounding)

**Used in**: Report 4 → Page 2 → Section 1 → Field #1 (Bucket % per competitor)

---

##### Function 7: `calculate_high_volume_markers()`

**Function**: `calculate_high_volume_markers(bucket_percentages: dict) -> dict`

**Purpose**: Flag buckets with >20% allocation (high volume focus) for visual marking

**Type**: 🆕 Entirely new function

**Reuses**: Output from `calculate_bucket_distribution()` (Function 6)

**Input Parameters**:
- `bucket_percentages` (dict): Output from calculate_bucket_distribution()
  - Example: `{"0-3s": 8, "3-9s": 36, "9-13s": 18, ...}`

**Process**:
1. Set threshold: 20% (1.6x uniform distribution across 8 buckets)
2. For each bucket:
   - Check if `percentage > 20`
   - Flag as True if yes, False if no
3. Return dict with 8 boolean flags

**Example Implementation**:
```python
def calculate_high_volume_markers(bucket_percentages: dict) -> dict:
    """
    Flag buckets with >20% allocation (high volume focus).

    Args:
        bucket_percentages: Output from calculate_bucket_distribution()

    Returns:
        dict: {bucket_name: is_high_volume (bool)}

    Example:
        >>> bucket_pct = {"0-3s": 8, "3-9s": 36, "9-13s": 18, ...}
        >>> calculate_high_volume_markers(bucket_pct)
        {
            "0-3s": False,
            "3-9s": True,   # 36% > 20%
            "9-13s": False,
            "13-18s": False,
            "18-33s": False,
            "33-60s": False,
            "60-90s": False,
            "90-120s": False
        }
    """
    high_volume_threshold = 20  # >20% = high volume

    markers = {}
    for bucket, percentage in bucket_percentages.items():
        markers[bucket] = percentage > high_volume_threshold

    return markers
```

**Output**: Dict with 8 boolean values (e.g., `{"0-3s": False, "3-9s": True, ...}`)

**Display Logic**: If `True`, append " 🟢" to percentage in report table

**Verified Example**: Drinkpoppi 3-9s = True (36% > 20%), all others = False

**Used in**: Report 4 → Page 2 → Section 1 → Field #2 (High volume markers)

**Threshold Rationale**: 20% = 1.6x uniform distribution (12.5% per bucket), indicates significant content focus

---

##### Function 8: `calculate_market_patterns()`

**Function**: `calculate_market_patterns(all_competitors_distributions: dict) -> dict`

**Purpose**: Categorize market volume level per bucket based on average across all competitors

**Type**: 🆕 Entirely new function

**Reuses**: Output from `calculate_bucket_distribution()` for all competitors (Function 6)

**Input Parameters**:
- `all_competitors_distributions` (dict): Dict mapping competitor handles to bucket distributions
  - Example: `{"@nike": {"0-3s": 2, "3-9s": 5, ...}, "@adidas": {"0-3s": 3, ...}, ...}`

**Process**:
1. For each of 8 buckets:
   - Extract percentages from all competitors
   - Calculate average: `avg = sum(percentages) / len(competitors)`
   - Categorize based on thresholds:
     - ≥25% → "HIGH VOLUME"
     - 20-24% → "High volume"
     - 15-19% → "Moderate volume"
     - 10-14% → "Growing volume"
     - <10% → "Low volume"
2. Return dict with 8 category labels

**Example Implementation**:
```python
def calculate_market_patterns(all_competitors_distributions: dict) -> dict:
    """
    Categorize market volume level per bucket.

    Args:
        all_competitors_distributions: Dict of {competitor_handle: bucket_distribution}

    Returns:
        dict: {bucket_name: market_pattern_label}

    Example:
        >>> competitors = {
        ...     "@nike": {"0-3s": 2, "3-9s": 5, "18-33s": 28, ...},
        ...     "@adidas": {"0-3s": 3, "3-9s": 8, "18-33s": 32, ...},
        ...     "@puma": {"0-3s": 5, "3-9s": 10, "18-33s": 26, ...}
        ... }
        >>> calculate_market_patterns(competitors)
        {
            "0-3s": "Low volume",         # Avg: 3.3%
            "3-9s": "Low volume",         # Avg: 7.7%
            "18-33s": "HIGH VOLUME",      # Avg: 28.7%
            ...
        }
    """
    all_buckets = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
    market_patterns = {}

    for bucket in all_buckets:
        # Calculate average percentage across all competitors
        percentages = [comp_dist[bucket] for comp_dist in all_competitors_distributions.values()]
        avg_percentage = sum(percentages) / len(percentages)

        # Categorize based on average
        if avg_percentage >= 25:
            pattern = "HIGH VOLUME"
        elif avg_percentage >= 20:
            pattern = "High volume"
        elif avg_percentage >= 15:
            pattern = "Moderate volume"
        elif avg_percentage >= 10:
            pattern = "Growing volume"
        else:
            pattern = "Low volume"

        market_patterns[bucket] = pattern

    return market_patterns
```

**Output**: Dict with 8 string labels (e.g., `{"0-3s": "Low volume", "18-33s": "HIGH VOLUME", ...}`)

**Categorization Thresholds**:
- **≥25%**: HIGH VOLUME (uppercase) - Primary battleground
- **20-24%**: High volume - Significant focus
- **15-19%**: Moderate volume - Medium investment
- **10-14%**: Growing volume - Emerging focus
- **<10%**: Low volume - Minimal presence

**Verified with Template**: Template examples match threshold logic (28.7% avg → "HIGH VOLUME", 23.3% → "High volume", etc.)

**Used in**: Report 4 → Page 2 → Section 1 → Field #3 (Market pattern per bucket)

---

##### Function 9: `get_unique_winning_buckets()`

**Function**: `get_unique_winning_buckets(client_id: str, competitors: list[str]) -> list[str]`

**Purpose**: Get union of top_3_buckets across all competitors for Report 4 Performance by Duration table

**Type**: 🆕 Entirely new function

**Reuses**: Data from `winner_analysis.json` (existing file)

**Context**: Report 4 shows performance metrics for each competitor's winning buckets. Each competitor has 3 winning buckets, but the union across all competitors may be 5-7 unique buckets. This function discovers all unique buckets to build the table rows.

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitors` (list[str]): List of competitor handles from CLI `--competitors` parameter
  - Example: `["@drinkpoppi", "@vitalproteins", "@nike"]`

**Process**:
1. Initialize empty set for unique buckets
2. For each competitor handle in list:
   - Remove `@` symbol from handle (e.g., `"@drinkpoppi"` → `"drinkpoppi"`)
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - Discover analysis directory (starts with `"top_"`, should be only one)
   - Load `{analysis_dir}/winner_analysis.json`
   - Extract `top_3_buckets` array
   - Add all buckets to union set (deduplicate automatically)
3. Sort by duration order: `["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]`
4. Return sorted list

**Example Implementation**:
```python
def get_unique_winning_buckets(client_id: str, competitors: list[str]) -> list[str]:
    """
    Get union of top_3_buckets across all competitors.
    Dynamically discovers the analysis directory per competitor.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitors: List of competitor handles with @ symbol (e.g., ["@nike", "@adidas"])

    Returns:
        list[str]: Sorted list of unique bucket names

    Example:
        >>> get_unique_winning_buckets("test_run", ["@drinkpoppi", "@vitalproteins"])
        ["3-9s", "9-13s", "13-18s", "18-33s", "33-60s"]  # 5 unique buckets from 2 competitors

    Note:
        - Each competitor has exactly 3 winning buckets (top_3_buckets)
        - Union may contain 3-7 unique buckets depending on overlap
        - Analysis directory is discovered dynamically (e.g., "top_contrastive" or "top_top")
    """
    import os
    import json
    import logging

    logger = logging.getLogger(__name__)
    all_buckets = set()

    for competitor_handle in competitors:
        # Remove @ symbol for directory name
        competitor_dir = competitor_handle.lstrip('@')

        # Base path for this competitor
        base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

        # Discover analysis directory (should be only one starting with "top_")
        try:
            analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

            if not analysis_dirs:
                logger.warning(
                    f"No analysis directory found for {competitor_handle} at {base_path}"
                )
                continue

            # Use first directory (should be only one)
            analysis_dir = analysis_dirs[0]

            # Load winner_analysis.json
            winner_path = f"{base_path}/{analysis_dir}/winner_analysis.json"

            if not os.path.exists(winner_path):
                logger.warning(f"winner_analysis.json not found: {winner_path}")
                continue

            with open(winner_path, 'r') as f:
                data = json.load(f)

            # Extract top_3_buckets
            top_3 = data.get('top_3_buckets', [])

            # Add to union set (deduplicates automatically)
            all_buckets.update(top_3)

        except Exception as e:
            logger.error(
                f"Error processing competitor {competitor_handle}: {str(e)}"
            )
            continue

    # Sort by duration order
    bucket_order = [
        "0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"
    ]
    sorted_buckets = sorted(
        all_buckets,
        key=lambda b: bucket_order.index(b) if b in bucket_order else 99
    )

    return sorted_buckets
```

**Output**: List of bucket names sorted by duration (e.g., `["3-9s", "9-13s", "13-18s", "18-33s", "33-60s"]`)

**Edge Cases**:
- **No overlap**: 3 competitors × 3 buckets each = 9 unique buckets → unlikely but possible
- **Complete overlap**: All competitors share same 3 buckets = 3 unique buckets → common for similar content strategies
- **Missing analysis**: If competitor has no `winner_analysis.json`, skip with warning
- **Different strategies**: Competitor directories may use `top_contrastive` or `top_top` → function discovers dynamically

**Validation Status**: ✅ **VERIFIED** with actual data
- **drinkpoppi**: `top_3_buckets = ["3-9s", "9-13s", "18-33s"]`
- **vitalproteins**: `top_3_buckets = ["13-18s", "33-60s", "9-13s"]`
- **Union**: `["3-9s", "9-13s", "13-18s", "18-33s", "33-60s"]` (5 unique buckets)

**Used in**: Report 4 → Page 2 → Section 2: Performance by Duration → Field #1 (table rows)

---

##### Function 10: `calculate_competitor_bucket_avg_views()`

**Function**: `calculate_competitor_bucket_avg_views(client_id: str, competitor_handle: str, bucket_name: str) -> Optional[int]`

**Purpose**: Calculate average views for a specific bucket (if it's a winning bucket), returns None if not

**Type**: 🔄 Wrapper function (adaptation of Function 1)

**Reuses**: ✅ Similar logic to `calculate_competitor_avg_views()` but returns per-bucket average

**Context**: Performance by Duration table (Report 4 Page 2 Section 2) shows avg views per bucket per competitor. Each cell needs individual bucket averages, not overall average. Returns None/"—" if bucket not in competitor's top_3_buckets.

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitor_handle` (str): Competitor handle with @ symbol
  - Example: `"@nike"`
- `bucket_name` (str): Duration bucket to calculate for
  - Example: `"13-18s"`

**Process**:
1. **Discover analysis directory dynamically**:
   - Remove `@` from handle: `"@nike"` → `"nike"`
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - List directories starting with `"top_"` (e.g., `"top_contrastive"` or `"top_top"`)
   - Use first directory found (should be only one)
2. Load `winner_analysis.json → top_3_buckets`
3. **Check if bucket is a winning bucket**:
   - If `bucket_name` NOT in `top_3_buckets`: return `None` (display as "—")
   - If `bucket_name` IN `top_3_buckets`: continue to step 4
4. Load `buckets/bucket_{bucket_name}/selected_videos.json`
5. Get first `top_count` videos (top performers)
6. Calculate average: `sum(playCount) / top_count`
7. Return integer

**Example Implementation**:
```python
def calculate_competitor_bucket_avg_views(
    client_id: str,
    competitor_handle: str,
    bucket_name: str
) -> Optional[int]:
    """
    Calculate average views for a specific bucket (if it's a winning bucket).
    Dynamically discovers the analysis directory per competitor.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitor_handle: Competitor handle with @ symbol (e.g., "@nike")
        bucket_name: Duration bucket (e.g., "13-18s")

    Returns:
        int: Average views for this bucket (e.g., 580000), or None if not a winning bucket

    Example:
        >>> calculate_competitor_bucket_avg_views("test_run", "@nike", "13-18s")
        580000  # 580K avg views for 13-18s bucket

        >>> calculate_competitor_bucket_avg_views("test_run", "@nike", "60-90s")
        None  # 60-90s not in @nike's top_3_buckets
    """
    import os
    import json
    from typing import Optional

    # Remove @ symbol for directory name
    competitor_dir = competitor_handle.lstrip('@')

    # Base path
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

    # Discover analysis directory (should be only one starting with "top_")
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(
            f"No analysis directory found for {competitor_handle} at {base_path}"
        )

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]

    # Check if this bucket is a winning bucket
    if bucket_name not in winning_buckets:
        return None  # Display as "—" in report

    # Load selected videos for this bucket
    bucket_path = f"{competitor_path}/buckets/bucket_{bucket_name}/selected_videos.json"

    if not os.path.exists(bucket_path):
        raise FileNotFoundError(f"selected_videos.json not found: {bucket_path}")

    with open(bucket_path) as f:
        data = json.load(f)

    top_count = data["top_count"]
    top_videos = data["videos"][:top_count]  # First N videos = top performers

    # Calculate bucket average
    avg_views = sum(v["playCount"] for v in top_videos) / len(top_videos)

    return int(avg_views)
```

**Output**:
- Integer (e.g., 580000) if bucket is a winning bucket → format with K/M suffix (580K)
- `None` if bucket not in top_3_buckets → display as "—"

**Display Logic**:
```python
views = calculate_competitor_bucket_avg_views(client_id, "@nike", "13-18s")
display_value = format_number_with_suffix(views) if views is not None else "—"
# Result: "580K" or "—"
```

**Used in**: Report 4 → Page 2 → Section 2: Performance by Duration → Field #2 (Avg views per competitor per bucket)

**Relationship to Function 1**:
- **Function 1** (`calculate_competitor_avg_views`): Returns weighted average across ALL 3 winning buckets → used in Performance Rankings table
- **Function 10** (`calculate_competitor_bucket_avg_views`): Returns average for ONE specific bucket (or None) → used in Performance by Duration table

---

##### Function 11: `calculate_competitor_bucket_avg_engagement()`

**Function**: `calculate_competitor_bucket_avg_engagement(client_id: str, competitor_handle: str, bucket_name: str) -> Optional[float]`

**Purpose**: Calculate average engagement rate for a specific bucket (if it's a winning bucket), returns None if not

**Type**: 🔄 Wrapper function (adaptation of Function 4)

**Reuses**: ✅ Calls existing `calculate_engagement_metrics()` (Section 0.5.5)

**Context**: Performance by Duration table (Report 4 Page 2 Section 2) shows avg engagement per bucket per competitor. Each cell needs individual bucket engagement averages. Returns None/"—" if bucket not in competitor's top_3_buckets.

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitor_handle` (str): Competitor handle with @ symbol
  - Example: `"@nike"`
- `bucket_name` (str): Duration bucket to calculate for
  - Example: `"13-18s"`

**Process**:
1. **Discover analysis directory dynamically**:
   - Remove `@` from handle: `"@nike"` → `"nike"`
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - List directories starting with `"top_"` (e.g., `"top_contrastive"` or `"top_top"`)
   - Use first directory found (should be only one)
2. Load `winner_analysis.json → top_3_buckets`
3. **Check if bucket is a winning bucket**:
   - If `bucket_name` NOT in `top_3_buckets`: return `None` (display as "—")
   - If `bucket_name` IN `top_3_buckets`: continue to step 4
4. Load `buckets/bucket_{bucket_name}/selected_videos.json`
5. Get first `top_count` videos (top performers)
6. For each video:
   - Load `analysis/unified_analysis/{video_id}.json → metadata`
   - Extract engagement fields: `playCount, diggCount, commentCount, shareCount, collectCount`
   - Call existing `calculate_engagement_metrics(metadata)` ← BASE FUNCTION
   - Collect `engagement_rate`
7. Calculate average: `sum(engagement_rates) / len(engagement_rates)`
8. Return float (rounded to 2 decimals)

**Example Implementation**:
```python
def calculate_competitor_bucket_avg_engagement(
    client_id: str,
    competitor_handle: str,
    bucket_name: str
) -> Optional[float]:
    """
    Calculate average engagement rate for a specific bucket (if it's a winning bucket).
    Dynamically discovers the analysis directory per competitor.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitor_handle: Competitor handle with @ symbol (e.g., "@nike")
        bucket_name: Duration bucket (e.g., "13-18s")

    Returns:
        float: Average engagement % for this bucket (e.g., 1.4), or None if not a winning bucket

    Example:
        >>> calculate_competitor_bucket_avg_engagement("test_run", "@nike", "13-18s")
        1.4  # 1.4% avg engagement for 13-18s bucket

        >>> calculate_competitor_bucket_avg_engagement("test_run", "@nike", "60-90s")
        None  # 60-90s not in @nike's top_3_buckets
    """
    import os
    import json
    from typing import Optional

    # Remove @ symbol for directory name
    competitor_dir = competitor_handle.lstrip('@')

    # Base path
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

    # Discover analysis directory (should be only one starting with "top_")
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(
            f"No analysis directory found for {competitor_handle} at {base_path}"
        )

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]

    # Check if this bucket is a winning bucket
    if bucket_name not in winning_buckets:
        return None  # Display as "—" in report

    # Load selected videos for this bucket
    bucket_path = f"{competitor_path}/buckets/bucket_{bucket_name}/selected_videos.json"

    if not os.path.exists(bucket_path):
        raise FileNotFoundError(f"selected_videos.json not found: {bucket_path}")

    with open(bucket_path) as f:
        data = json.load(f)

    top_count = data["top_count"]
    top_videos = data["videos"][:top_count]  # First N videos = top performers

    all_engagement_rates = []

    for video in top_videos:
        video_id = video["id"]

        # Load unified analysis metadata
        unified_path = f"{competitor_path}/buckets/bucket_{bucket_name}/analysis/unified_analysis/{video_id}.json"

        if not os.path.exists(unified_path):
            # Skip videos with missing unified analysis
            continue

        with open(unified_path) as f:
            unified_data = json.load(f)

        metadata = unified_data["metadata"]

        # ← CALL EXISTING BASE FUNCTION (Section 0.5.5)
        engagement_result = calculate_engagement_metrics(metadata)
        engagement_rate = engagement_result["engagement_rate"]

        all_engagement_rates.append(engagement_rate)

    # Calculate average engagement
    if not all_engagement_rates:
        return None  # No valid data

    avg_engagement = round(sum(all_engagement_rates) / len(all_engagement_rates), 2)

    return avg_engagement
```

**Output**:
- Float (e.g., 1.4) if bucket is a winning bucket → display as "1.4%"
- `None` if bucket not in top_3_buckets → display as "—"

**Display Logic**:
```python
engagement = calculate_competitor_bucket_avg_engagement(client_id, "@nike", "13-18s")
display_value = f"{engagement}%" if engagement is not None else "—"
# Result: "1.4%" or "—"
```

**Used in**: Report 4 → Page 2 → Section 2: Performance by Duration → Field #3 (Avg engagement per competitor per bucket)

**Relationship to Function 4**:
- **Function 4** (`calculate_competitor_avg_engagement`): Returns average engagement across ALL 3 winning buckets → used in Performance Rankings table
- **Function 11** (`calculate_competitor_bucket_avg_engagement`): Returns average for ONE specific bucket (or None) → used in Performance by Duration table

**Data Fields Used**: `metadata.{playCount, diggCount, commentCount, shareCount, collectCount}`

---

##### Function 12: `calculate_bucket_best_performer()`

**Function**: `calculate_bucket_best_performer(client_id: str, competitors: list[str], bucket_name: str) -> str`

**Purpose**: Determine which competitor performs best in a specific bucket using composite score (views + engagement)

**Type**: 🆕 Entirely new function

**Reuses**: ✅ Calls Functions 10 & 11, similar logic to Function 5 (market leader)

**Context**: Performance by Duration table (Report 4 Page 2 Section 2) shows best performer per bucket. Each row needs to identify which competitor has highest composite score for that duration bucket.

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitors` (list[str]): List of all competitor handles
  - Example: `["@nike", "@adidas", "@puma"]`
- `bucket_name` (str): Duration bucket to analyze
  - Example: `"13-18s"`

**Process**:
1. Initialize list to collect competitor data for this bucket
2. For each competitor:
   - Call `calculate_competitor_bucket_avg_views(client_id, competitor, bucket_name)` → Function 10
   - If None (not a winning bucket), skip this competitor
   - Call `calculate_competitor_bucket_avg_engagement(client_id, competitor, bucket_name)` → Function 11
   - If None, skip this competitor
   - Store: `{"handle": competitor, "views": views, "engagement": engagement}`
3. **If no competitors have data for this bucket**: return `"N/A"` or `"—"`
4. Calculate composite scores (same as Function 5 logic):
   - Find max_views across competitors with data
   - For each competitor: `composite_score = (views / max_views × 100) + engagement`
5. Find competitor(s) with highest composite score
6. **Handle ties**: If multiple competitors have same composite score, use tie-breaking logic:
   - Check if tied on both views AND engagement → add " (tie)"
   - Check if views differ but engagement wins → add " (engagement wins tie)"
   - Check if engagement differs but views win → add " (views wins tie)"
7. Return best competitor handle with optional tie notation

**Example Implementation**:
```python
def calculate_bucket_best_performer(
    client_id: str,
    competitors: list[str],
    bucket_name: str
) -> str:
    """
    Determine which competitor performs best in a specific bucket.
    Uses composite score (normalized views + engagement).

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitors: List of all competitor handles (e.g., ["@nike", "@adidas", "@puma"])
        bucket_name: Duration bucket (e.g., "13-18s")

    Returns:
        str: Best performer handle with optional tie notation
            Examples: "@nike", "@adidas (engagement wins tie)", "—"

    Example:
        >>> calculate_bucket_best_performer("test_run", ["@nike", "@adidas"], "13-18s")
        "@nike"  # Nike has highest composite score in 13-18s

        >>> calculate_bucket_best_performer("test_run", ["@nike", "@adidas"], "60-90s")
        "—"  # Neither competitor has 60-90s in their top_3_buckets
    """
    import logging

    logger = logging.getLogger(__name__)

    # Step 1: Collect data for competitors who have this bucket
    competitors_with_data = []

    for competitor in competitors:
        # Get views for this bucket (None if not a winning bucket)
        views = calculate_competitor_bucket_avg_views(client_id, competitor, bucket_name)

        if views is None:
            continue  # Skip competitors without this bucket

        # Get engagement for this bucket
        engagement = calculate_competitor_bucket_avg_engagement(client_id, competitor, bucket_name)

        if engagement is None:
            continue  # Skip if engagement data missing

        competitors_with_data.append({
            "handle": competitor,
            "views": views,
            "engagement": engagement
        })

    # Step 2: Handle case where no competitors have data
    if not competitors_with_data:
        return "—"

    # Step 3: Calculate composite scores
    max_views = max(c["views"] for c in competitors_with_data)

    for competitor in competitors_with_data:
        # Normalize views to 0-100 scale
        views_score = (competitor["views"] / max_views) * 100
        # Engagement already in % (0-100)
        engagement_score = competitor["engagement"]
        # Composite score
        competitor["composite_score"] = views_score + engagement_score

    # Step 4: Find best competitor(s)
    max_score = max(c["composite_score"] for c in competitors_with_data)
    best_competitors = [c for c in competitors_with_data if c["composite_score"] == max_score]

    # Step 5: Handle ties
    if len(best_competitors) == 1:
        return best_competitors[0]["handle"]

    # Multiple competitors with same composite score
    # Check if it's a perfect tie (same views AND engagement)
    first = best_competitors[0]
    if all(c["views"] == first["views"] and c["engagement"] == first["engagement"]
           for c in best_competitors):
        # Perfect tie - return first alphabetically with "(tie)" notation
        best_competitors.sort(key=lambda c: c["handle"])
        return f"{best_competitors[0]['handle']} (tie)"

    # Tie in composite but different views/engagement
    # Find who has better engagement among tied competitors
    max_engagement = max(c["engagement"] for c in best_competitors)
    engagement_winners = [c for c in best_competitors if c["engagement"] == max_engagement]

    if len(engagement_winners) == 1:
        return f"{engagement_winners[0]['handle']} (engagement wins tie)"

    # Still tied - return first alphabetically with engagement wins notation
    engagement_winners.sort(key=lambda c: c["handle"])
    return f"{engagement_winners[0]['handle']} (engagement wins tie)"
```

**Output**: String with competitor handle and optional tie notation
- `"@nike"` - Clear winner
- `"@adidas (tie)"` - Perfect tie (same views and engagement)
- `"@nike (engagement wins tie)"` - Composite score tied, engagement higher
- `"—"` - No competitors have this bucket in top_3

**Tie-Breaking Logic**:
1. **Perfect tie** (same views + engagement): Add " (tie)"
2. **Composite tie, different metrics**: Winner determined by engagement → Add " (engagement wins tie)"
3. **No data**: Return "—"

**Used in**: Report 4 → Page 2 → Section 2: Performance by Duration → Field #5 (Best performer per bucket)

**Relationship to Function 5**:
- **Function 5** (`calculate_market_leader`): Determines overall market leader across all buckets
- **Function 12** (`calculate_bucket_best_performer`): Determines best performer for ONE specific bucket

---

##### Function 13: `is_winning_bucket()`

**Function**: `is_winning_bucket(client_id: str, competitor_handle: str, bucket_name: str) -> bool`

**Purpose**: Check if a bucket is in a competitor's top_3_buckets (for winning bucket marker 👑)

**Type**: 🆕 Entirely new function (helper function)

**Reuses**: Same path discovery logic as Functions 9-12

**Context**: Performance by Duration table needs to display "👑" marker for each competitor's winning buckets. This function provides a testable, debuggable way to check bucket membership.

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitor_handle` (str): Competitor handle with @ symbol
  - Example: `"@nike"`
- `bucket_name` (str): Duration bucket to check
  - Example: `"13-18s"`

**Process**:
1. **Discover analysis directory dynamically**:
   - Remove `@` from handle: `"@nike"` → `"nike"`
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - List directories starting with `"top_"` (e.g., `"top_contrastive"` or `"top_top"`)
   - Use first directory found (should be only one)
2. Load `winner_analysis.json → top_3_buckets`
3. Check if `bucket_name in top_3_buckets`
4. Return boolean result

**Example Implementation**:
```python
def is_winning_bucket(client_id: str, competitor_handle: str, bucket_name: str) -> bool:
    """
    Check if bucket is in competitor's top_3_buckets.
    Dynamically discovers the analysis directory per competitor.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitor_handle: Competitor handle with @ symbol (e.g., "@nike")
        bucket_name: Duration bucket (e.g., "13-18s")

    Returns:
        bool: True if bucket is in top_3_buckets, False otherwise

    Example:
        >>> is_winning_bucket("test_run", "@nike", "13-18s")
        True  # 13-18s is in Nike's top 3

        >>> is_winning_bucket("test_run", "@nike", "60-90s")
        False  # 60-90s not in Nike's top 3
    """
    import os
    import json

    # Remove @ symbol for directory name
    competitor_dir = competitor_handle.lstrip('@')

    # Base path
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

    # Discover analysis directory (should be only one starting with "top_")
    try:
        analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

        if not analysis_dirs:
            raise FileNotFoundError(
                f"No analysis directory found for {competitor_handle} at {base_path}"
            )

        analysis_dir = analysis_dirs[0]
        competitor_path = f"{base_path}/{analysis_dir}"

        # Load winning buckets
        with open(f"{competitor_path}/winner_analysis.json") as f:
            winner_data = json.load(f)

        top_3_buckets = winner_data["top_3_buckets"]

        # Check membership
        return bucket_name in top_3_buckets

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error checking winning bucket for {competitor_handle}: {str(e)}")
        return False
```

**Output**: Boolean (`True` or `False`)

**Display Logic**:
```python
marker = "👑" if is_winning_bucket(client_id, competitor, bucket) else ""
```

**Used in**: Report 4 → Page 2 → Section 2: Performance by Duration → Field #4 (Winning bucket markers)

**Benefits**:
- ✅ Testable with unit tests
- ✅ Centralized error handling
- ✅ Consistent with other functions
- ✅ Easy to add logging/debugging

---

##### Function 14: `get_competitor_winning_buckets()`

**Function**: `get_competitor_winning_buckets(client_id: str, competitor_handle: str) -> list[str]`

**Purpose**: Get the list of top 3 winning buckets for a competitor

**Type**: 🆕 Entirely new function (helper function)

**Reuses**: Same path discovery logic as Functions 9-13

**Context**: Performance by Duration section displays each competitor's winning bucket list. This function provides a testable, debuggable way to retrieve the top_3_buckets array.

**Input Parameters**:
- `client_id` (str): Client identifier
  - Example: `"test_run"`
- `competitor_handle` (str): Competitor handle with @ symbol
  - Example: `"@nike"`

**Process**:
1. **Discover analysis directory dynamically**:
   - Remove `@` from handle: `"@nike"` → `"nike"`
   - Construct base path: `/data/clients/{client_id}/competitors/{competitor_dir}/`
   - List directories starting with `"top_"` (e.g., `"top_contrastive"` or `"top_top"`)
   - Use first directory found (should be only one)
2. Load `winner_analysis.json → top_3_buckets`
3. Return array of bucket names

**Example Implementation**:
```python
def get_competitor_winning_buckets(client_id: str, competitor_handle: str) -> list[str]:
    """
    Get list of top 3 winning buckets for a competitor.
    Dynamically discovers the analysis directory per competitor.

    Args:
        client_id: Client identifier (e.g., "test_run")
        competitor_handle: Competitor handle with @ symbol (e.g., "@nike")

    Returns:
        list[str]: List of 3 bucket names (e.g., ["9-13s", "13-18s", "18-33s"])

    Example:
        >>> get_competitor_winning_buckets("test_run", "@nike")
        ["9-13s", "13-18s", "18-33s"]  # Nike's top 3 performing buckets
    """
    import os
    import json

    # Remove @ symbol for directory name
    competitor_dir = competitor_handle.lstrip('@')

    # Base path
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"

    # Discover analysis directory (should be only one starting with "top_")
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(
            f"No analysis directory found for {competitor_handle} at {base_path}"
        )

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)

    return winner_data["top_3_buckets"]
```

**Output**: List of 3 bucket names (e.g., `["9-13s", "13-18s", "18-33s"]`)

**Display Logic**:
```python
buckets = get_competitor_winning_buckets(client_id, competitor)
display_value = ", ".join(buckets)  # "9-13s, 13-18s, 18-33s"

# Or bullet list format:
for bucket in buckets:
    print(f"• {bucket}")
```

**Used in**: Report 4 → Page 2 → Section 2: Performance by Duration → Field #6 (Competitor winning bucket lists)

**Benefits**:
- ✅ Testable with unit tests
- ✅ Centralized error handling
- ✅ Consistent with other functions
- ✅ Easy to add logging/debugging
- ✅ Single source of truth for top_3_buckets data

**Relationship to Function 9**:
- **Function 9** (`get_unique_winning_buckets`): Returns UNION of top_3_buckets across ALL competitors
- **Function 14** (`get_competitor_winning_buckets`): Returns top_3_buckets for ONE specific competitor

---

**Function Summary for Report 4**:

| # | Function | Type | Reuses | Purpose |
|---|----------|------|--------|---------|
| **Performance Rankings Functions** |
| 1 | `calculate_competitor_avg_views()` | 🆕 New | Data structure | Weighted avg views (all buckets) |
| 2 | `calculate_posting_frequency()` | 🆕 New | Data files | Videos per week |
| 3 | `calculate_videos_analyzed()` | 🆕 New | Data structure | Sum selected count |
| 4 | `calculate_competitor_avg_engagement()` | 🔄 Wrapper | ✅ `calculate_engagement_metrics()` | Avg engagement |
| 5 | `calculate_market_leader()` | 🆕 New | Report 1 pattern | Composite ranking |
| **Bucket Distribution Functions** |
| 6 | `calculate_bucket_distribution()` | 🆕 New | winner_analysis.json | Bucket percentages |
| 7 | `calculate_high_volume_markers()` | 🆕 New | Function 6 output | High volume flags |
| 8 | `calculate_market_patterns()` | 🆕 New | Function 6 (all comps) | Market categorization |
| **Performance by Duration Functions** |
| 9 | `get_unique_winning_buckets()` | 🆕 New | winner_analysis.json | Union of top_3_buckets |
| 10 | `calculate_competitor_bucket_avg_views()` | 🔄 Wrapper | Function 1 logic | Avg views per bucket |
| 11 | `calculate_competitor_bucket_avg_engagement()` | 🔄 Wrapper | Function 4 + `calculate_engagement_metrics()` | Avg engagement per bucket |
| 12 | `calculate_bucket_best_performer()` | 🆕 New | Functions 10 & 11 | Best performer per bucket |
| 13 | `is_winning_bucket()` | 🆕 New | winner_analysis.json | Check bucket membership (helper) |
| 14 | `get_competitor_winning_buckets()` | 🆕 New | winner_analysis.json | Get top_3_buckets per competitor |

**Function Removed**:
- ~~`generate_key_insights()`~~: Removed due to edge case issues with diverse competitor strategies

**Implementation Status**: All 14 functions require implementation for Report 4 generation.

---

### Section 1: Designer Templates (8 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 1.1 | Design Template A (Content Creator) | Designer | 2 days | 2-page, mobile-optimized, includes 2 QR code placeholders |
| 1.2 | Design Template B (Client Executive) | Designer | 2 days | 3-page, intelligence dashboard |
| 1.3 | Design Template C (Single Competitor) | Designer | 2 days | 3-page, benchmarking vs client |
| 1.4 | Design Template D (Comparison) | Designer | 2 days | 4-page, side-by-side multi-competitor |

**Template A Requirements** (from Issue 1 resolution - see Stage8MVP_Reports.md):
- Include 2 QR code placeholders (~1" x 1" each):
  - **QR Code 1**: After "The Proof" section (links to top performer video)
  - **QR Code 2**: In "Contrastive Analysis" section (links to bottom performer video)
- Labels: "Scan to watch: Top Performer Using This Pattern (520K views)" and "Bottom Performer - Don't Do This (95K views)"
- Technical: Leave square placeholder boxes for QR code image insertion during manual workflow

**Deliverables**: 4 editable PDF templates (InDesign/Canva/Figma) with clearly labeled text boxes

**Template Requirements**:
- **Labeled placeholders** for all data fields (e.g., `[PATTERN_NAME]`, `[AVG_VIEWS]`, `[ENGAGEMENT_DIFFERENCE]`)
- **Mobile-optimized** (Template A): Min 12pt body, 16pt+ headings, portrait layout
- **Chart templates**: Pre-designed bar charts, star ratings, timeline graphics
- **Brand consistency**: Tumi Labs colors, fonts, logo placement

---

### Section 2: Branding Package (3 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 2.1 | Create visual identity system | Designer | 1 day | Colors, fonts, spacing, grids |
| 2.2 | Create chart templates | Designer | 1 day | Bar charts, star ratings, timelines |
| 2.3 | Create icon library + assets | Designer | 1 day | Logos, dividers, backgrounds |

**Deliverables**:
- Brand style guide (PDF)
- Chart template library (editable files)
- Asset package (PNG/SVG files)

---

#### 0.5.7: Visual Direction Categorization

**Function**: `get_visual_direction(avg_eye_contact_rate: float, avg_face_size: float) -> str`

**Purpose**: Categorize visual framing/direction based on eye contact and face size metrics from temporal windows

**When to Use**:
- Report 2 (Hashtag → Creator): Phase 1 Hook execution guidance
- Any report needing visual framing descriptions for creator guidance

**Input Parameters**:
- `avg_eye_contact_rate` (float): Average eye contact rate from hook window (0-3s) across top performers
  - Range: 0.0 to 1.0
  - Source: `temporal_windows.hook.eye_contact_rate`
- `avg_face_size` (float): Average face size ratio from hook window across top performers
  - Range: 0.0 to 1.0 (proportion of frame)
  - Source: `temporal_windows.hook.average_face_size`

**Process**:
1. Check if high eye contact (>0.7) AND large face size (>0.3) → Close-up, direct
2. Else if high eye contact (>0.7) → Direct to camera, medium shot
3. Else if large face size (>0.3) → Face visible but not direct
4. Else → Wide shot or object-focused

**Example Implementation**:
```python
def get_visual_direction(avg_eye_contact_rate: float, avg_face_size: float) -> str:
    """
    Categorize visual framing based on eye contact and face size.

    Args:
        avg_eye_contact_rate: Average eye contact rate (0.0-1.0)
        avg_face_size: Average face size ratio (0.0-1.0)

    Returns:
        str: Visual direction description for creator guidance

    Example:
        >>> get_visual_direction(0.87, 0.44)
        "Face visible, direct to camera (close-up)"

        >>> get_visual_direction(0.90, 0.05)
        "Face visible, direct to camera (medium shot)"

        >>> get_visual_direction(0.0, 0.44)
        "Face visible, not direct to camera"

        >>> get_visual_direction(0.0, 0.05)
        "Wide shot or object-focused"
    """
    # High eye contact + close face = direct close-up
    if avg_eye_contact_rate > 0.7 and avg_face_size > 0.3:
        return "Face visible, direct to camera (close-up)"

    # High eye contact but distant = direct medium shot
    elif avg_eye_contact_rate > 0.7:
        return "Face visible, direct to camera (medium shot)"

    # Face present but no eye contact
    elif avg_face_size > 0.3:
        return "Face visible, not direct to camera"

    # Low values = wide or object-focused
    else:
        return "Wide shot or object-focused"
```

**Output Format**:
```python
# String describing visual framing for creator execution guidance
"Face visible, direct to camera (close-up)"
```

**Usage in Reports**:
- **Template 2 (Creator)**: Phase 1 Hook execution guidance
- Shows creators exact framing/camera positioning needed

**Data Source**:
- Temporal windows data: `{video_id}_temporal_windows_updated.json` → `temporal_windows.hook`
- Validated with real data from 5 sample videos (eye_contact range: 0.0-0.90, face_size range: 0.0-0.44)

---

### Section 3: Data Extraction Scripts (3.25 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 3.1 | Build `extract_creator_data.py` + QR generation | Developer | 1.25 days | Stage 7 → 9 creator datasets + 18 QR code images |
| 3.2 | Build `extract_client_data.py` | Developer | 1 day | Stages 1,6,7 → client dashboard data (Google Sheets) |
| 3.3 | Build `extract_competitor_data.py` | Developer | 1 day | Competitor analysis → benchmarking data (Google Sheets) |

**QR Code Addition** (from Issue 1 resolution):
- Task 3.1 now includes QR code generation (+0.25 days effort)
- Generates 2 QR codes per formula (18 total per hashtag: 9 formulas × 2 codes)
- Maps Stage 2 video URLs (top/bottom cluster) to Stage 7 formulas
- Uses Python `qrcode` library to generate PNG files
- Output: `{hashtag}_{bucket}_{formula}_top.png` and `_bottom.png`

**Script Requirements**:
- **Input**: JSON files from Stages 1, 6, 7 (existing outputs)
- **Output**: ✅ **Google Sheets** with clearly labeled sections and data fields
- **Google Sheets API**: Use `gspread` or `google-sheets-python-api` for sheet creation
- **Sheet Structure**: One sheet per report type, with formatted headers and data rows
- **Error handling**: Clear error messages if JSON missing/malformed
- **CLI interface**: Simple command-line invocation

---

#### 3.1: `extract_creator_data.py`

**Purpose**: Extract 9 creative formulas from Stage 7 JSON → formatted datasets

**Input**:
- Stage 7 `winning_formulas.json` (3 winning buckets × 3 formulas each)

**Output Format** (per formula):
```
=== FORMULA 1: 18-33s Bucket ===
Pattern Name: [The Question Hook Formula]
Duration: [18-33s]
Hashtag: [#nutrition]
Confidence: [87%]

--- THE PROOF ---
PERFORMANCE COMPARISON:

Videos using this pattern (Top Cluster):
• Average Views: [620K]
• Average Engagement: [1.2%] ([7,440] interactions/video)

Videos NOT using this pattern (Bottom Cluster):
• Average Views: [380K]
• Average Engagement: [0.8%] ([3,040] interactions/video)

RESULTS:
→ [1.6x] MORE VIEWS (63% higher reach)
→ [1.5x] MORE ENGAGEMENT (50% higher resonance)

--- CONTRASTIVE ANALYSIS ---
Top performers do THIS:
✅ [Ask question in first 2s (avg 3.2 questions in hook)]
✅ [Show product by 5 seconds (immediate visual payoff)]
✅ [Use 5-7 text overlays (keep attention with text)]

Bottom performers do THIS:
❌ [Generic opening/statement (0.8 questions avg)]
❌ [Product reveal after 10+ seconds (viewers already scrolled)]
❌ [No text overlays (viewers get bored/confused)]

--- PATTERN SUMMARY ---
1️⃣ Hook (0-3s): [Ask compelling question]
2️⃣ Show (3-15s): [Reveal product + explain benefit]
3️⃣ Prove (15-33s): [Demonstrate result + CTA]

--- SECOND-BY-SECOND TIMELINE ---
⏱️ 0-2 seconds: THE QUESTION
  Say: [Did you know [surprising fact about topic]?]
  Visual: [Your face, direct to camera]
  Text overlay: [The question (animated in)]

⏱️ 3-5 seconds: SHOW THE THING
  Visual: [Close-up of product/ingredient]
  Text overlay: [Product name]
  Say: [This is [product name]]

[... continues for all timestamps ...]

--- PRE-POST CHECKLIST ---
□ [Question in first 2 seconds?]
□ [Product visible by 5 seconds?]
□ [5-7 text overlays placed?]
□ [2-3 scene changes in middle?]
□ [Clear CTA at end?]
```

**CLI Usage**:
```bash
python extract_creator_data.py --client acme --hashtag nutrition --mode top --strategy engagement
```

**Output**: Google Sheet created with URL logged to console and saved to:
`/data/clients/acme/hashtags/nutrition/top_engagement/extracted_creator_reports_sheet_url.txt`

**Google Sheet Structure**:
- **Sheet Name**: "nutrition_creator_reports"
- **9 tabs**: One per formula (Formula_1_18-33s, Formula_2_18-33s, ..., Formula_9_60-90s)
- **Each tab structure**: Sections as rows with labeled headers (Pattern Name, The Proof, Contrastive Analysis, etc.)

---

#### 3.2: `extract_client_data.py`

**Purpose**: Extract hashtag intelligence dashboard data for client executive report

**Input**:
- Cluster Config: `/config/hashtag_clusters/{target}.json` (primary hashtag)
- Cluster Analytics: `/data/clients/{client}/hashtag/{target}/cluster_analytics.json` (total scraped videos, duration stats)
- Winner Analysis: `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/winner_analysis.json` (top 3 buckets, percentages)
- Selection Manifest: `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/selection_manifest.json` (top/bottom performer counts)
- Stage 6 `rf_video_analysis.json` (ML metrics)
- Stage 7 `winning_formulas.json` (formula names for Page 3)

**Output Format**:

Google Sheet with 3 tabs containing data for client executive report:

**Tab 1: Page_1_Scale_of_Analysis**
- Hashtag (from cluster config)
- Analysis period (static text)
- Total videos analyzed (from cluster analytics)
- Winning buckets (3) with percentages (from winner analysis)
- Top/bottom performer counts (from selection manifest)
- Total video duration (calculated)
- Analysis method description (static text)

**Tab 2: Page_2_Hashtag_Intelligence**
- Duration distribution (8 buckets with percentages)
- Performance metrics (3 winning buckets: views + engagement)
- Content intelligence (categories, hooks, pain points, keywords from Stage 2.7)
- Creator profile priorities (tiered list based on winning buckets)

**Tab 3: Page_3_Your_Reports**
- Report distribution (9 formulas across 3 buckets)
- Formula names per bucket (from Stage 7 winning_formulas.json)
- Report contents description (static text)

**For detailed report template structure and dynamic field mappings, see:**
→ `Stage8MVP_Reports.md` Section "1. Hashtag → Client (Executive Report)"

**CLI Usage**:
```bash
python extract_client_data.py --client acme --hashtag nutrition --mode top --strategy engagement
```

**Output**: Google Sheet created with URL logged to console and saved to:
`/data/clients/acme/hashtags/nutrition/top_engagement/extracted_client_dashboard_sheet_url.txt`

**Google Sheet Structure**:
- **Sheet Name**: "nutrition_client_dashboard"
- **3 tabs**: Page_1_Scale_of_Analysis, Page_2_Hashtag_Intelligence, Page_3_Your_Reports
- **Each tab structure**: Sections as rows with labeled data fields

---

#### 3.3: `extract_competitor_data.py`

**Purpose**: Extract competitor benchmarking data for single or comparison reports

**Engagement Metrics**: Uses `calculate_engagement_metrics()` (Section 0.5.5) to calculate real engagement rates from Apify metadata for both competitor and client videos

**Input**:
- Competitor Stage 7 `winning_formulas.json`
- Competitor Stage 6 ML analysis JSONs
- Competitor Stage 1 `winner_analysis.json` (bucket distribution)
- Competitor Stage 2 metadata (`views`, `likes`, `comments`, `shares`, `saves`) for engagement calculation
- Competitor metadata (handle, posting frequency, top hashtags)
- Client baseline data (for benchmarking)

**Output Format (Single Competitor)**:
```
=== COMPETITOR OVERVIEW ===
Competitor Handle: [@rival_brand]
Client Baseline: [@acme_nutrition]
Analysis Period: [Past 2-3 months]

--- POSTING ACTIVITY ---
Competitor Posts (analyzed): [82 videos]
Client Posts (analyzed): [65 videos]
Posting Frequency Gap: [Competitor posts 26% more]

--- PERFORMANCE BENCHMARKING ---
Competitor Avg Views: [450K]
Client Avg Views: [320K]
Performance Gap: [Competitor gets 41% more views]

Competitor Avg Engagement: [6.2%]
Client Avg Engagement: [4.8%]
Engagement Gap: [Competitor gets 29% more engagement]

--- TOP BUCKETS COMPARISON ---
Competitor Top Bucket: [18-33s (35% of content)]
Client Top Bucket: [13-18s (28% of content)]

Competitor #2 Bucket: [13-18s (22% of content)]
Client #2 Bucket: [18-33s (25% of content)]

--- CREATIVE PATTERNS (Competitor Top 3 Formulas) ---
Formula 1: [The Ingredient Shock Hook]
  Bucket: [18-33s]
  Performance: [8.1% avg engagement]

Formula 2: [The Product Transformation Demo]
  Bucket: [18-33s]
  Performance: [7.8% avg engagement]

Formula 3: [The Quick Win Tutorial]
  Bucket: [13-18s]
  Performance: [7.4% avg engagement]

--- HASHTAG STRATEGY ---
Competitor Top Hashtags:
  1. [#nutrition (45% of posts)]
  2. [#healthylifestyle (38% of posts)]
  3. [#wellness (32% of posts)]

Client Top Hashtags:
  1. [#nutrition (62% of posts)]
  2. [#fitness (28% of posts)]
  3. [#healthtips (22% of posts)]

Insight: [Competitor diversifies hashtags more, client over-indexes on #nutrition]

--- OPPORTUNITIES ---
Gap 1: [Competitor dominates 18-33s bucket - client should invest here]
Gap 2: [Competitor uses shock hooks effectively - client lacks this pattern]
Gap 3: [Competitor hashtag diversification strategy outperforms client focus]
```

**Output Format (Comparison Report - Multi-Competitor)**:
```
=== MULTI-COMPETITOR COMPARISON ===
Client Baseline: [@acme_nutrition]
Competitors Analyzed: [@rival_brand, @competitor2, @competitor3]

--- PERFORMANCE LEADERBOARD ---
Rank 1: [@rival_brand] - [450K avg views, 6.2% engagement]
Rank 2: [@competitor2] - [410K avg views, 5.8% engagement]
Rank 3: [@acme_nutrition] - [320K avg views, 4.8% engagement] ← CLIENT
Rank 4: [@competitor3] - [280K avg views, 4.1% engagement]

--- BUCKET STRATEGY COMPARISON ---
                    | Client  | Rival   | Comp2   | Comp3   |
--------------------|---------|---------|---------|---------|
Top Bucket          | 13-18s  | 18-33s  | 18-33s  | 13-18s  |
% in Top Bucket     | 28%     | 35%     | 40%     | 25%     |
Top Bucket Avg Views| 520K    | 480K    | 510K    | 390K    |

Insight: [Rival and Comp2 dominate 18-33s with high volume - client should shift focus]

--- CREATIVE FORMULA COMPARISON ---
Best Performer Overall: [@rival_brand - The Ingredient Shock Hook (8.1% engagement)]
Client Best Formula: [The Question Hook Formula (5.2% engagement)]
Gap: [Client's best formula underperforms market leader by 2.9 percentage points]

--- HASHTAG STRATEGY COMPARISON ---
Most Diverse: [@rival_brand (uses 12 hashtags regularly)]
Least Diverse: [@acme_nutrition (uses 5 hashtags regularly)]
Recommendation: [Expand hashtag portfolio to reach broader audience]

--- KEY TAKEAWAYS ---
1. [Client ranks 3rd out of 4 in performance - 41% gap vs leader]
2. [Shift content focus to 18-33s bucket (where top 2 competitors dominate)]
3. [Adopt shock hook formula similar to @rival_brand's approach]
4. [Diversify hashtag strategy - currently too narrow]
```

**CLI Usage**:
```bash
# Single competitor
python extract_competitor_data.py --client acme --competitor rival_brand

# Comparison (multiple competitors)
python extract_competitor_data.py --client acme --competitors rival_brand,competitor2,competitor3
```

**Output**: Google Sheets created with URLs logged to console and saved to:
- Single: `/data/clients/acme/competitors/rival_brand/top_engagement/extracted_competitor_sheet_url.txt`
- Comparison: `/data/clients/acme/competitors/comparison_sheet_url_2025-01-28.txt`

**Google Sheet Structure (Single)**:
- **Sheet Name**: "rival_brand_vs_acme"
- **1 tab**: Single sheet with sections as rows (Competitor Overview, Performance Benchmarking, Creative Patterns, Hashtag Strategy, Opportunities)

**Google Sheet Structure (Comparison)**:
- **Sheet Name**: "acme_competitor_comparison"
- **1 tab**: Single sheet with comparison tables (Performance Leaderboard, Bucket Strategy Comparison, Formula Comparison, Hashtag Strategy, Key Takeaways)

---

### Section 4: Documentation (1 day)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 4.1 | Write extraction script instructions | Developer | 0.5 days | CLI usage, troubleshooting |
| 4.2 | Create template population guide | Designer + Developer | 0.5 days | Step-by-step copy-paste workflow |

**Deliverables**:
- `STAGE8_EXTRACTION_GUIDE.md` - How to run scripts, interpret output
- `TEMPLATE_POPULATION_GUIDE.md` - How to populate each template with extracted data
- Video tutorial (optional): 10-min walkthrough of full workflow

---

### Section 5: Testing (0.5 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 5.1 | Test extraction scripts on real data | Developer | 0.25 days | Run on actual Stage 1-7 outputs |
| 5.2 | Generate 1 sample PDF of each type | Designer + Developer | 0.25 days | Validate template + data integration |

**Test Cases**:
- Extract data from `#nutrition` hashtag analysis (output to Google Sheets)
- Open Google Sheets and verify data accuracy
- Populate all 4 templates from Google Sheets data
- Export PDFs and verify mobile rendering (Template A)
- Confirm all placeholders populated correctly

---

## Total MVP Effort: ~17 days

| Section | Tasks | Effort |
|---------|-------|--------|
| Section 0: Template Structures | 4 tasks | 0 days (Tasks 0.1, 0.2 ✅ COMPLETE, 0.3, 0.4 remaining) |
| Section 1: Designer Templates | 4 tasks | 8 days (includes QR code placeholders in Template A) |
| Section 2: Branding Package | 3 tasks | 3 days |
| Section 3: Data Extraction Scripts (Google Sheets) | 3 tasks | 3.25 days (includes QR code generation) |
| Section 4: Documentation | 2 tasks | 1 day |
| Section 5: Testing | 2 tasks | 0.5 days |
| **TOTAL** | **18 tasks** | **15.75 days (~3 weeks)** (2 tasks complete, 16 remaining) |

**Scope Changes from Issue Resolutions**:
- Task 0.2 ✅ COMPLETE: Hashtag → Creator template (Stage8MVP_Reports.md Section 2)
- Task 3.1 updated: +0.25 days for QR code generation (Issue 1: Visual Examples)
- Template A updated: Includes 2 QR code placeholders per report

**Parallelizable**: Designer work (Sections 1-2: 11 days) + Development work (Section 3: 3.25 days) can run simultaneously after Section 0 complete

**Critical Path**:
1. Section 0 (0.75 days remaining: Tasks 0.3, 0.4) → BLOCKS everything
2. Section 1-2 (11 days) designer work in parallel with Section 3 (3 days) dev work
3. Section 4-5 (1.5 days) sequential after above

**Actual Calendar Time**: ~2.5 weeks (if parallelized) to ~3.5 weeks (if sequential)

---

## Phased Approach Recommendation

### Phase 1 (NOW): Designer Template MVP
**Timeline**: 3.5 weeks
**Effort**: 16.75 days development (1.25 days template structures + 3 days scripts + 11 days designer + 1.5 days docs/testing)
**Use for**: Onboarding (5 hashtags, 5-7 competitors)
**Manual time investment**: ~25 hours total onboarding + ~2 hrs/month ongoing

**Why start here**:
- Get to production 10x faster
- Validate report content and design with real clients
- Learn what clients actually want before automating

---

### Phase 2 (IF NEEDED): Partial Automation
**Trigger**: Clients request weekly reports OR 5+ active clients
**Timeline**: 2 weeks
**Focus**: Automate ONLY the most repetitive part (Creator PDFs - 9 per hashtag)

**What to automate**:
- Creator report generation (Template A) - saves ~3 hours per hashtag
- Keep client + competitor reports manual (less frequent)

**Development**: ~10 days
**Savings**: ~15 hours/month (if running 5 hashtag analyses/month)

---

### Phase 3 (SCALE): Full Automation
**Trigger**: 10+ active clients OR 20+ hours/month manual work
**Timeline**: 8 weeks
**Scope**: Implement full Stage8Planning.md

**What to automate**:
- All 4 report types (creator, client, single competitor, comparison)
- Full PDF generation engine
- Batch processing, error handling, version control

**Development**: ~50 days (original automated MVP)
**Savings**: All manual work eliminated

---

## Required Resources

### Software Licenses
- **Adobe InDesign** (~$30/month) OR **Canva Pro** (~$13/month) OR **Figma** (free tier OK)
- **Python 3.8+** (free)
- ✅ **Google Workspace** (REQUIRED - for Google Sheets API access)

### Skills Needed
- **Designer**: Adobe InDesign/Canva proficiency, brand identity design
- **Developer**: Python, JSON parsing, basic data transformation
- **You**: Willingness to spend ~25 hours on manual work during onboarding

### Time Commitment
- **Upfront**: 3 weeks (designer + dev work in parallel)
- **Onboarding**: ~25 hours manual work across 5 hashtags + 7 competitors
- **Ongoing**: ~30 min every 2 weeks (biweekly client reports)

---

## Success Criteria

### MVP Complete When:
1. ✅ All 4 designer templates finalized and tested
2. ✅ All 3 extraction scripts working on real Stage 1-7 data
3. ✅ 1 sample PDF generated for each template type
4. ✅ Documentation complete (extraction + template guides)
5. ✅ Mobile rendering validated for Template A (creator reports)

### Production-Ready When:
1. ✅ 5 hashtag analyses completed (45 creator PDFs + 5 client PDFs generated)
2. ✅ 5 single competitor reports generated
3. ✅ 2 comparison reports generated
4. ✅ Client feedback collected and templates iterated
5. ✅ Team trained on extraction + population workflow

---

## Next Steps

### Immediate Actions (Week 1)
1. ✅ **COMPLETE Section 0 first** - Create 2 remaining template structures (0.3, 0.4) before anything else
2. ✅ Select design software (InDesign vs Canva vs Figma)
3. ✅ Hire/assign designer
4. ⏸️ **WAIT for Section 0** - Designer cannot start until all template structures complete
5. ⏸️ **WAIT for Section 0** - Developer can start Section 3 after Section 0 complete

### Week 2
1. ✅ Designer completes Templates A + B
2. ✅ Developer completes scripts 3.1 + 3.2
3. ✅ Test extraction scripts on real data
4. ✅ Generate first sample PDFs (creator + client)

### Week 3
1. ✅ Designer completes Templates C + D
2. ✅ Developer completes script 3.3
3. ✅ Generate sample PDFs (competitor single + comparison)
4. ✅ Write documentation (extraction + population guides)
5. ✅ Final testing and validation

### Week 4+ (Production Use)
1. ✅ Run first onboarding hashtag analysis → generate 9 creator + 1 client PDFs
2. ✅ Collect client feedback
3. ✅ Iterate on templates if needed
4. ✅ Continue onboarding (remaining 4 hashtags + 7 competitors)

---

## Open Questions for Decision

1. ✅ **Output Format** - RESOLVED: Google Sheets (easiest to review/edit)
2. **Design Software**: Adobe InDesign, Canva Pro, or Figma? (Affects designer workflow and license cost)
3. **Template D Priority**: Should we defer Comparison Report template to Phase 2? (Only needed 2 times during onboarding vs 5 times for single competitor)
4. **Quality Assurance**: Who will review PDFs before sending to clients? (Peer review vs self-review)

**Recommendation**: Answer questions 2-4 before starting development to avoid rework.

---

## Appendix: File Structure

### Extraction Script Outputs

```
/data/clients/{client_id}/
├── hashtags/{hashtag}/{mode}_{strategy}/
│   ├── extracted_creator_reports_sheet_url.txt    # Google Sheets URL (Output of 3.1)
│   ├── extracted_client_dashboard_sheet_url.txt   # Google Sheets URL (Output of 3.2)
│   └── final_reports/
│       ├── nutrition_18-33s_formula_1.pdf         # Manually generated from Template A
│       ├── nutrition_18-33s_formula_2.pdf
│       ├── ... (9 creator PDFs total)
│       └── nutrition_client_report.pdf            # Manually generated from Template B
│
└── competitors/{competitor_handle}/{mode}_{strategy}/
    ├── extracted_competitor_sheet_url.txt         # Google Sheets URL (Output of 3.3 single)
    ├── comparison_sheet_url_2025-01-28.txt        # Google Sheets URL (Output of 3.3 comparison)
    └── final_reports/
        ├── rival_brand_vs_acme_intel.pdf          # Manually generated from Template C
        └── acme_competitor_comparison.pdf         # Manually generated from Template D
```

**Note**: Google Sheets URLs are also logged to console during script execution for easy access

### Designer Template Files

```
/design_assets/
├── templates/
│   ├── Template_A_ContentCreator_v1.indd      # InDesign source
│   ├── Template_B_ClientExecutive_v1.indd
│   ├── Template_C_SingleCompetitor_v1.indd
│   └── Template_D_Comparison_v1.indd
│
├── brand_assets/
│   ├── TumiLabs_Logo.svg
│   ├── icon_library/
│   ├── chart_templates/
│   └── brand_style_guide.pdf
│
└── documentation/
    ├── STAGE8_EXTRACTION_GUIDE.md
    └── TEMPLATE_POPULATION_GUIDE.md
```

---

**Status**: ✅ **READY TO START** - Section 0 (template structures) is the critical path blocker. Complete Tasks 0.3, 0.4 before designer/dev work begins.

**Next Action**: Create Handle/Single Competitor → Client template structure (Task 0.3)
