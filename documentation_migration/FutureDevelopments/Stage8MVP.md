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

**Phase 2: Data Extraction Scripts** (can happen in parallel with designer - 3.25 days):
1. `extract_creator_data.py` - Stage 7 JSON → 3 formatted creator reports (Excel file with 3 tabs)
2. `extract_client_data.py` - Stages 1,6,7 → 1 client executive dashboard (Excel file)
3. `extract_competitor_data.py` - Single competitor analysis → benchmarking data (Excel file)
4. `extract_multi_competitor_data.py` - Multi-competitor (2-5) → market intelligence (Excel file)

**Output Format**: ✅ **Excel files (.xlsx)** - Simple, offline, no authentication required

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

### Hashtag Analysis → Content Creators (3 PDFs)

**Frequency**: Onboarding (~5 times), then rarely needed

| Step | Who | Time | Details |
|------|-----|------|---------|
| 1. Run pipeline Stages 1-7 | Automated | Auto | Existing ML pipeline |
| 2. Extract data + QR codes | Script | 30 sec | `python extract_creator_data.py --hashtag nutrition` → Excel file (3 tabs, 1 per formula) + 6 QR PNGs |
| 3. Review data | You | 15 min | Open Excel, verify accuracy, edit if needed |
| 4. Populate Template A (x3) | You | ~1 hr | Copy-paste from Excel + insert 2 QR code images per report (~20 min each) |
| 5. Export PDFs | You | 5 min | Save as PDF from InDesign/Canva |

**Total Manual Time per Hashtag**: ~1.5 hours (for 3 creator PDFs, includes QR code insertion)

**Onboarding Total**: ~7.5 hours across 5 hashtags

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
- Report 3 (Single Competitor): Aggregate competitor's content strategy patterns
- Report 4 Section 4 (Caption Strategy): Aggregate caption metrics (CTA type, hashtag count) for multi-competitor comparison

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
   - **Caption Strategy Fields** (3): `cta_type`, `hook_type`, `hashtag_count` (mean/min/max/median)
   - **Caption Fields NOT USED IN REPORTS** (3): `emoji_usage`, `caption_length`, `hashtag_placement` (excluded per Report 2 design decision - low actionability for creators)
5. Calculate effect sizes (if both top and bottom groups aggregated)

**Note on `confidence` field**: Used for filtering (Step 3), NOT included in aggregated output. This ensures only reliable classifications inform reports.

**Field Selection Rationale**: The 9 aggregated fields were chosen based on ContentAnalysisCHILDpt2.md Decision 2 (80/20 rule - highest value fields for actionable insights).

**Excluded Caption Fields**:
- `caption_hook_type` → Kept as `hook_type` (used in Report 2 CAPTION STRUCTURE)
- `emoji_usage` → Excluded (low specificity: just "some" vs "many")
- `caption_length` → Excluded (binary "short" vs "long", not actionable)
- `hashtag_placement` → Excluded (low variance, "end" always wins)
- `brand_mention_present` → Excluded (niche-specific, not generalizable)
- `influencer_tag_present` → Excluded (depends on collaboration availability)

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

#### 0.5.1.2: Taxonomy Description Lookup

**Function**: `get_descriptions_from_taxonomy(category_names: list[str], taxonomy_type: str) -> list[str]`

**Purpose**: Look up human-readable descriptions for classification categories from Stage 2.6 taxonomy

**Why This Exists**:
- Stage 2.7 returns category names in snake_case (e.g., `"recipe_tutorial"`)
- Reports need human-readable descriptions (e.g., `"Step-by-step cooking instructions"`)
- Stage 2.6 maintains the taxonomy with descriptions for each category
- This function provides the lookup/mapping layer

**Prerequisites**:
- Stage 2.6 must create taxonomy JSON files with category descriptions
- Taxonomy files should be stored at: `/config/taxonomies/{taxonomy_type}.json`

**Input Parameters**:
- `category_names` (list[str]): List of category names from Stage 2.7
  - Example: `["recipe_tutorial", "wellness_practice", "supplement_review"]`
- `taxonomy_type` (string): Type of taxonomy to lookup
  - Valid values: `"content_category"`, `"hook_strategy"`, `"engagement_drivers"`, `"content_tactics"`, `"pain_points"`

**Returns**:
- Array of human-readable descriptions matching the input order
- Example: `["Step-by-step cooking instructions", "Daily health routines and habits", "Product recommendations and reviews"]`

**Taxonomy File Format**:
```json
{
  "recipe_tutorial": {
    "name": "Recipe Tutorial",
    "description": "Step-by-step cooking instructions"
  },
  "wellness_practice": {
    "name": "Wellness Practice",
    "description": "Daily health routines and habits"
  },
  "supplement_review": {
    "name": "Supplement Review",
    "description": "Product recommendations and reviews"
  }
}
```

**Example Implementation**:
```python
import json
from typing import List

def get_descriptions_from_taxonomy(category_names: List[str], taxonomy_type: str) -> List[str]:
    """
    Look up descriptions from taxonomy files.

    Args:
        category_names: List of snake_case category names
        taxonomy_type: Type of taxonomy (e.g., "content_category")

    Returns:
        List of human-readable descriptions
    """
    # Load taxonomy file
    taxonomy_path = f"/config/taxonomies/{taxonomy_type}.json"

    try:
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
    except FileNotFoundError:
        # Fallback: Convert snake_case to Title Case if taxonomy not found
        return [name.replace("_", " ").title() for name in category_names]

    # Look up descriptions
    descriptions = []
    for name in category_names:
        if name in taxonomy:
            descriptions.append(taxonomy[name]["description"])
        else:
            # Fallback for missing entries
            descriptions.append(name.replace("_", " ").title())

    return descriptions
```

**Example Usage**:
```python
# Get content category descriptions
categories = ["recipe_tutorial", "wellness_practice", "supplement_review"]
descriptions = get_descriptions_from_taxonomy(categories, "content_category")
# Returns: ["Step-by-step cooking instructions", "Daily health routines...", "Product recommendations..."]

# Get hook strategy descriptions
hooks = ["question_hook", "problem_solution", "direct_statement"]
descriptions = get_descriptions_from_taxonomy(hooks, "hook_strategy")
# Returns: ["Opens with engaging question", "Identifies pain point...", "Bold claim or fact"]
```

**Used in**:
- Report 1 → Page 3 → Section 1 (Content category descriptions)
- Report 2 → Multiple sections (Hook strategy descriptions, etc.)
- Report 3 → Page 3 → Section 1 (Content category descriptions)
- Report 3 → Page 3 → Section 2 (Hook strategy descriptions)

**Error Handling**:
- If taxonomy file doesn't exist → Fallback to title-cased snake_case conversion
- If category not in taxonomy → Fallback to title-cased snake_case conversion
- Always returns a list matching input length

**Note**: This function depends on Stage 2.6 taxonomy creation. Until Stage 2.6 is implemented, the fallback behavior (title-casing snake_case) will be used.

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

**Function**: `extract_hashtag_analysis(client_id, competitor_handle, mode="top", strategy="contrastive")`

**Purpose**: Extract hashtag analytics from competitor's selected videos across winning buckets

**When to Use**:
- Report 3 (Competitor): Show top hashtags competitor uses
- Report 4 (Multi-Competitor): Compare hashtag strategies across competitors (Section 3)

**Input Parameters**:
- `client_id` (string): Client identifier
- `competitor_handle` (string): Competitor handle (e.g., "@drinkpoppi")
- `mode` (string, optional): Analysis mode (default: "top")
- `strategy` (string, optional): Selection strategy (default: "contrastive")

**Returns**:
```python
{
    'total_unique_hashtags': 42,           # Count of distinct hashtags across all winning buckets
    'avg_hashtags_per_video': 11,          # Mean hashtag count per video
    'total_videos_analyzed': 59,           # Sum of top performers across 3 winning buckets
    'top_10_hashtags': [                   # Top 10 by usage frequency
        {
            'tag': '#wellness',
            'usage_pct': 78,                # (videos with tag / total videos) × 100%
            'video_count': 46
        },
        # ... 9 more
    ],
    'top_5_concentration': 65              # % of total hashtag occurrences from top 5 (shows strategic focus: >70% = focused, <70% = diversified)
}
```

**Data Source**: `selected_videos.json` from each winning bucket
- **Path**: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/buckets/bucket_{name}/selected_videos.json`
- **Field**: `videos[].hashtags[].name`
- **Videos Used**: Top performers only (first `top_count` videos per bucket)

**Process**:
1. Load `winner_analysis.json` to identify 3 winning buckets
2. For each winning bucket:
   - Load `selected_videos.json`
   - Extract hashtags from top performers: `videos[0:top_count].hashtags[].name`
3. Aggregate across all 3 buckets:
   - Count unique hashtags
   - Calculate usage percentages
   - Rank by frequency
4. Calculate metrics:
   - Total unique hashtags
   - Average hashtags per video
   - Top 5 concentration percentage

**Example Implementation**:
```python
def extract_hashtag_analysis(client_id, competitor_handle, mode="top", strategy="contrastive"):
    """Extract hashtag analytics from competitor's winning buckets."""
    from collections import Counter

    # Construct base path with dynamic discovery
    base_path = f"/data/clients/{client_id}/competitors/{competitor_handle}/"

    # Find strategy directory dynamically
    dirs = [d for d in os.listdir(base_path) if d.startswith(f'{mode}_')]
    strategy_dir = dirs[0]  # Use discovered directory

    competitor_path = f"{base_path}/{strategy_dir}"

    # Load winner_analysis to get winning buckets
    winner_analysis = json.load(open(f"{competitor_path}/winner_analysis.json"))
    winning_buckets = winner_analysis['top_3_buckets']

    # Collect all hashtags from top performers
    all_hashtags = []
    total_videos = 0

    for bucket in winning_buckets:
        selected_videos_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"
        data = json.load(open(selected_videos_path))

        top_count = data['top_count']
        total_videos += top_count

        # Extract hashtags from top performers only
        for video in data['videos'][:top_count]:
            video_hashtags = [h['name'] for h in video['hashtags']]
            all_hashtags.extend(video_hashtags)

    # Calculate metrics
    hashtag_counter = Counter(all_hashtags)
    unique_count = len(hashtag_counter)
    avg_per_video = len(all_hashtags) / total_videos if total_videos > 0 else 0

    # Top 10 with usage percentages
    top_10 = [
        {
            'tag': f"#{tag}",
            'usage_pct': round((count / total_videos) * 100),
            'video_count': count
        }
        for tag, count in hashtag_counter.most_common(10)
    ]

    # Top 5 concentration: % of total hashtag occurrences from top 5
    # This measures strategic focus (high % = focused, low % = diversified)
    total_occurrences = sum(hashtag_counter.values())
    top_5_occurrences = sum(count for _, count in hashtag_counter.most_common(5))
    top_5_concentration = round((top_5_occurrences / total_occurrences) * 100) if total_occurrences > 0 else 0

    return {
        'total_unique_hashtags': unique_count,
        'avg_hashtags_per_video': round(avg_per_video),
        'total_videos_analyzed': total_videos,
        'top_10_hashtags': top_10,
        'top_5_concentration': top_5_concentration
    }
```

**Usage in Reports**:
- **Report 3**: `Stage8MVP_Reports.md` Section 3 (lines 1276-1312)
- **Report 4**: `Stage8MVP_Reports.md` Section 3 Hashtag Strategy Comparison (lines 2614-2668)

---

#### 0.5.4: @Mention Extraction

**Function**: `extract_mention_analysis(manifest_path)`

**Purpose**: Extract @mentions to identify affiliate/repost partnerships using two-filter detection approach

**When to Use**:
- Report 3 (Single Competitor): Analyze competitor's content sourcing strategy (original vs reposted)
- Report 4 (Multi-Competitor): Compare content sourcing strategies across competitors

**Input Parameters**:
- `manifest_path` (string): Path to `selection_manifest.json` file
  - Example: `/data/clients/{client}/competitors/{handle}/top_{strategy}/selection_manifest.json`

**Output Format**:
```python
{
    'top_10_mentions': [
        {'handle': '@fitnessguru123', 'mention_count': 54, 'percentage': 18.0},
        {'handle': '@healthcoach_jane', 'mention_count': 36, 'percentage': 12.0},
        # ... up to 10 handles
    ],
    'total_unique_mentions': 47,
    'videos_with_mentions': 82,
    'mention_rate': 93.2,
    'videos_with_repost_indicators': 37,
    'repost_rate': 42.0  # ← KEY FIELD for reports
}
```

**Source Data**: `selected_videos.json` → `videos[].text` (caption text)

**Two-Filter Repost Detection Logic**:

1. **Filter 1 (Primary): @Mention Presence**
   - If caption contains ANY @mention → classify as reposted/affiliate
   - Rationale: Brands mentioning other handles = affiliate/partnership content
   - Regex: `re.search(r'@\w+', caption)`

2. **Filter 2 (Backup): Keyword Indicators**
   - If no @mention but contains explicit repost keywords
   - Keywords: ['repost', 'via', 'credit', 'by', 'from']
   - Catches edge cases where @ is missing but attribution is explicit

**Process**:
1. Load `selection_manifest.json` to get selected video IDs
2. Extract all video IDs from `videos_by_bucket` (top + bottom performers across all 3 winning buckets)
3. For each video ID, load caption from `selected_videos.json` → `videos[].text`
4. Extract @mentions using regex: `re.findall(r'@(\w+)', caption)`
5. Apply two-filter detection:
   - Check Filter 1: Has @mentions?
   - If no, check Filter 2: Has repost keywords?
6. Aggregate @mention frequency counter across all selected videos
7. Calculate top 10 most-mentioned handles
8. Calculate repost rate: `(videos_with_reposts / total_videos) × 100%`

**Example Implementation**:
```python
import re
from collections import Counter
import json

def extract_mention_analysis(manifest_path):
    """
    Extract @mention analysis with two-filter repost detection.

    Args:
        manifest_path: Path to selection_manifest.json

    Returns:
        Dict with top_10_mentions, total_unique_mentions, repost_rate, etc.
    """
    # Load manifest to get selected video IDs
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    selected_video_ids = []
    for bucket, videos in manifest['videos_by_bucket'].items():
        selected_video_ids.extend(videos.get('top_performers', []))
        selected_video_ids.extend(videos.get('bottom_performers', []))

    # Load selected_videos.json to get captions
    bucket_path = manifest_path.replace('/selection_manifest.json', '/buckets')

    mention_counter = Counter()
    videos_with_mentions = 0
    videos_with_reposts = 0
    repost_indicators = ['repost', 'via', 'credit', 'by', 'from']

    for video_id in selected_video_ids:
        # Find video caption from selected_videos.json in any bucket
        caption = None
        for bucket in manifest['selected_buckets']:
            selected_videos_path = f"{bucket_path}/bucket_{bucket}/selected_videos.json"
            try:
                with open(selected_videos_path, 'r') as f:
                    selected_data = json.load(f)
                    for video in selected_data['videos']:
                        if video['id'] == video_id:
                            caption = video.get('text', '')
                            break
                if caption:
                    break
            except FileNotFoundError:
                continue

        if not caption:
            continue

        # Extract @mentions using regex (TikTok handle format)
        mentions = re.findall(r'@(\w+)', caption)

        if mentions:
            videos_with_mentions += 1
            mention_counter.update(mentions)

        # TWO-FILTER REPOST DETECTION
        is_repost = False

        # Filter 1 (Primary): @mention presence
        if mentions:
            is_repost = True

        # Filter 2 (Backup): Keyword indicators
        else:
            caption_lower = caption.lower()
            if any(indicator in caption_lower for indicator in repost_indicators):
                is_repost = True

        if is_repost:
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

**Usage Example**:
```python
# For Report 3 (Single Competitor)
manifest_path = "/data/clients/test_run/competitors/drinkpoppi/top_contrastive/selection_manifest.json"
results = extract_mention_analysis(manifest_path)

print(f"Repost Rate: {results['repost_rate']}%")
print(f"Total Unique @Mentions: {results['total_unique_mentions']}")
print(f"Top Affiliate: {results['top_10_mentions'][0]['handle']} ({results['top_10_mentions'][0]['percentage']}%)")

# Output:
# Repost Rate: 42.0%
# Total Unique @Mentions: 47
# Top Affiliate: @fitnessguru123 (18.0%)
```

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

##### Function 0.5.6.1: `calculate_avg_views_per_bucket()`

**Function**: `calculate_avg_views_per_bucket(bucket_path: str, performance_group: str = "top") -> int`

**Purpose**: Calculate average playCount for videos in a single bucket and performance group. This is the base function used across Reports 1, 2, and 3 for per-bucket view metrics.

**Type**: 🆕 New base function

**Relationship to Function 1 (0.5.6)**:
- **This function (0.5.6.1)**: Returns average views for ONE specific bucket
- **Function 1 (0.5.6)**: Calls this function 3 times (once per winning bucket) and calculates weighted average

**When to Use**:
- Report 1 (Hashtag → Client): Performance by Duration (Field #2), Creator Profile Priorities (Field #2)
- Report 2 (Hashtag → Creator): The Proof section (Fields #1, #4)
- Report 3 (Single Competitor): Performance metrics per bucket

**Input Parameters**:
- `bucket_path` (str): Absolute path to bucket folder
  - Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/`
- `performance_group` (str, optional): Filter by performance tier (default: "top")
  - Valid values: `"top"`, `"bottom"`, or `None` (all videos)
  - Most reports use `"top"` to show top performer averages

**Process**:
1. Load `{bucket_path}/selected_videos.json`
2. Extract `top_count` or `bottom_count` based on `performance_group`
3. Extract first N videos from `videos` array (pre-sorted by playCount DESC)
   - Top performers: `videos[0:top_count]`
   - Bottom performers: `videos[top_count:top_count+bottom_count]`
4. Calculate average: `sum(playCount) / count`
5. Return as integer

**Example Implementation**:
```python
def calculate_avg_views_per_bucket(bucket_path: str, performance_group: str = "top") -> int:
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

**Usage Examples**:

```python
# Report 1: Performance by Duration - Avg views for 18-33s bucket
avg_views_18_33 = calculate_avg_views_per_bucket(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/",
    performance_group="top"
)
# Returns: 1900000 (display as "1.9M")

# Report 2: The Proof - Top cluster average views
top_cluster_avg = calculate_avg_views_per_bucket(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_13-18s/",
    performance_group="top"
)
# Returns: 2100000 (display as "2.1M")

# Report 2: The Proof - Bottom cluster average views (for comparison)
bottom_cluster_avg = calculate_avg_views_per_bucket(
    bucket_path="/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_13-18s/",
    performance_group="bottom"
)
# Returns: 980000 (display as "980K")
```

**Data Source**:
```json
// From {bucket_path}/selected_videos.json
{
  "bucket": "18-33s",
  "strategy": "contrastive",
  "video_count": 100,
  "selected_count": 42,
  "top_count": 33,
  "bottom_count": 9,
  "videos": [
    // Sorted by playCount DESC
    {"id": "7540717847325003039", "playCount": 6700000, ...},  // Top performer #1
    {"id": "7539....", "playCount": 3200000, ...},             // Top performer #2
    // ... 31 more top performers
    {"id": "7522....", "playCount": 150000, ...},              // Bottom performer #1
    // ... 8 more bottom performers
  ]
}
```

**Verified Data Example**:
Using actual data from `/data/clients/test_competitor/competitors/drinkpoppi/top_contrastive/buckets/bucket_3-9s/`:
- Top count: 33 videos
- First video playCount: 6,700,000
- Calculation produces realistic averages matching Report 1 field examples

**Used In Reports**:

| Report | Section | Field | Line Reference |
|--------|---------|-------|----------------|
| Report 1 | Performance by Duration | Avg views per winning bucket (3 rows) | Stage8MVP_Reports.md:172 |
| Report 1 | Creator Profile Priorities | Avg views per winning bucket | Stage8MVP_Reports.md:199 |
| Report 2 | The Proof | Top cluster avg views | Stage8MVP_Reports.md:596 |
| Report 2 | The Proof | Bottom cluster avg views | Stage8MVP_Reports.md:598 |

**Error Handling**:
```python
# Invalid performance_group
calculate_avg_views_per_bucket(bucket_path, performance_group="invalid")
# Raises: ValueError("Invalid performance_group: invalid. Must be 'top', 'bottom', or None")

# Empty performance group (no videos)
calculate_avg_views_per_bucket(bucket_path_with_no_bottom_performers, performance_group="bottom")
# Returns: 0
```

**Integration with Function 1 (calculate_competitor_avg_views)**:
```python
# Function 1 (0.5.6) uses this function internally:
def calculate_competitor_avg_views(client_id, competitor_handle):
    # ... discover buckets ...

    bucket_stats = []
    for bucket in winning_buckets:
        bucket_path = f"{competitor_path}/buckets/bucket_{bucket}"

        # Calls this function (0.5.6.1)
        bucket_avg = calculate_avg_views_per_bucket(bucket_path, "top")

        bucket_stats.append({"avg_views": bucket_avg, "video_count": top_count})

    # Calculate weighted average across buckets
    weighted_avg = calculate_weighted_average(bucket_stats)
    return weighted_avg
```

---


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

#### 0.5.8: Bucket-Scoped Cluster Metrics (The Proof Section)

**Function**: `calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name, formula_cluster_id)`

**Purpose**: Calculate performance comparison metrics for "The Proof" section with bucket-scoping - ensures videos are filtered by BOTH cluster membership AND bucket duration

**When to Use**:
- Report 2 (Hashtag → Creator): "The Proof" section showing top cluster vs bottom cluster performance
- When you need to compare videos using a pattern vs not using it WITHIN a specific duration bucket

**Why Bucket-Scoping Matters**:
- Report 2 has 9 PDFs (3 buckets × 3 formulas)
- Each PDF is for ONE specific duration (e.g., "18-33s")
- "The Proof" should compare 18-33s videos using pattern vs 18-33s videos NOT using pattern
- Without bucket-scoping, metrics would mix all durations (9s, 18s, 33s, 60s), making comparison invalid for the specific duration

**Input Parameters**:
- `bucket_path` (string): Path to bucket folder
  - Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/`
- `bucket_name` (string): Duration bucket name
  - Example: `"18-33s"`, `"60-90s"`
  - Must match a bucket in `selection_manifest.json`
- `formula_cluster_id` (int): Winning cluster ID from Stage 7
  - Example: 0, 1, or 2 (from K-means clustering)

**Process**:
1. Load Stage 6 K-means cluster assignments to identify videos in winning cluster
2. Load `selection_manifest.json` to get top_performers for THIS BUCKET
3. Filter videos by: (1) cluster membership AND (2) bucket's top_performers
4. Calculate avg views and avg engagement for top cluster (bucket-scoped)
5. Calculate avg views and avg engagement for bottom cluster (bucket-scoped)
6. Calculate multipliers and percentage increases

**Example Implementation**:
```python
import json

def calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name, formula_cluster_id):
    """
    Calculate The Proof metrics with bucket-scoped cluster filtering.

    Returns performance comparison for videos in THIS BUCKET ONLY,
    comparing those using the pattern (winning cluster) vs not using it.
    """
    # Step 1: Load K-means cluster assignments
    kmeans_path = f"{bucket_path}/ml_analysis/hook_kmeans_analysis.json"
    with open(kmeans_path, 'r') as f:
        kmeans_data = json.load(f)

    # Get video IDs in winning cluster
    winning_cluster_video_ids = [
        v['video_id']
        for cluster in kmeans_data['clusters']
        if cluster['cluster_id'] == formula_cluster_id
        for v in cluster['videos']
    ]

    # Get video IDs NOT in winning cluster
    other_cluster_video_ids = [
        v['video_id']
        for cluster in kmeans_data['clusters']
        if cluster['cluster_id'] != formula_cluster_id
        for v in cluster['videos']
    ]

    # Step 2: Load selection manifest for THIS BUCKET
    manifest_path = f"{bucket_path}/../../selection_manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Get top performer IDs for THIS BUCKET ONLY (bucket-scoping!)
    bucket_top_performer_ids = manifest['videos_by_bucket'][bucket_name]['top_performers']

    # Step 3: Load selected videos
    selected_path = f"{bucket_path}/selected_videos.json"
    with open(selected_path, 'r') as f:
        selected_data = json.load(f)

    # Step 4: Filter for top cluster (bucket-scoped)
    # Videos must be: (1) in winning cluster AND (2) in this bucket's top performers
    top_cluster_videos = [
        v for v in selected_data['videos']
        if v['id'] in winning_cluster_video_ids
        and v['id'] in bucket_top_performer_ids
    ]

    # Step 5: Filter for bottom cluster (bucket-scoped)
    # Videos must be: (1) NOT in winning cluster AND (2) in this bucket's top performers
    bottom_cluster_videos = [
        v for v in selected_data['videos']
        if v['id'] in other_cluster_video_ids
        and v['id'] in bucket_top_performer_ids
    ]

    # Step 6: Calculate averages
    top_avg_views = sum(v['playCount'] for v in top_cluster_videos) / len(top_cluster_videos)
    bottom_avg_views = sum(v['playCount'] for v in bottom_cluster_videos) / len(bottom_cluster_videos)

    # Step 7: Calculate engagement (using calculate_engagement_metrics from Section 0.5.5)
    top_engagement_rates = []
    for v in top_cluster_videos:
        metadata = {
            'views': v['playCount'],
            'likes': v['diggCount'],
            'comments': v['commentCount'],
            'shares': v['shareCount'],
            'saves': v['collectCount']
        }
        metrics = calculate_engagement_metrics(metadata)
        top_engagement_rates.append(metrics['engagement_rate'])

    top_avg_engagement = sum(top_engagement_rates) / len(top_engagement_rates)

    bottom_engagement_rates = []
    for v in bottom_cluster_videos:
        metadata = {
            'views': v['playCount'],
            'likes': v['diggCount'],
            'comments': v['commentCount'],
            'shares': v['shareCount'],
            'saves': v['collectCount']
        }
        metrics = calculate_engagement_metrics(metadata)
        bottom_engagement_rates.append(metrics['engagement_rate'])

    bottom_avg_engagement = sum(bottom_engagement_rates) / len(bottom_engagement_rates)

    # Step 8: Calculate multipliers
    return {
        'top_cluster': {
            'avg_views': top_avg_views,
            'avg_engagement': top_avg_engagement,
            'video_count': len(top_cluster_videos)
        },
        'bottom_cluster': {
            'avg_views': bottom_avg_views,
            'avg_engagement': bottom_avg_engagement,
            'video_count': len(bottom_cluster_videos)
        },
        'multipliers': {
            'view_multiplier': round(top_avg_views / bottom_avg_views, 1),
            'engagement_multiplier': round(top_avg_engagement / bottom_avg_engagement, 1),
            'view_pct_increase': round(((top_avg_views - bottom_avg_views) / bottom_avg_views) * 100),
            'engagement_pct_increase': round(((top_avg_engagement - bottom_avg_engagement) / bottom_avg_engagement) * 100)
        }
    }
```

**Output Format**:
```python
{
    'top_cluster': {
        'avg_views': 620000,
        'avg_engagement': 1.2,
        'video_count': 25  # Only 18-33s videos using pattern
    },
    'bottom_cluster': {
        'avg_views': 380000,
        'avg_engagement': 0.8,
        'video_count': 15  # Only 18-33s videos NOT using pattern
    },
    'multipliers': {
        'view_multiplier': 1.6,
        'engagement_multiplier': 1.5,
        'view_pct_increase': 63,
        'engagement_pct_increase': 50
    }
}
```

**Usage Example**:
```python
# For Report 2 covering 18-33s bucket, Formula 1
bucket_path = "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s"
bucket_name = "18-33s"
formula_cluster_id = 0  # From winning_formulas.json

metrics = calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name, formula_cluster_id)

# Results are ONLY for 18-33s videos
print(f"Top cluster (18-33s using pattern): {metrics['top_cluster']['avg_views']:,} views")
print(f"Bottom cluster (18-33s NOT using pattern): {metrics['bottom_cluster']['avg_views']:,} views")
print(f"Multiplier: {metrics['multipliers']['view_multiplier']}x more views")
```

**Validation Status**: ⚠️ **AWAITING STAGE 6 + STAGE 7** (K-means clustering + winning formulas)

**Key Features**:
- ✅ Bucket-scoped (only videos from specified duration)
- ✅ Cluster-based (pattern users vs non-pattern users)
- ✅ Apples-to-apples comparison (same duration, different pattern usage)
- ✅ Uses real engagement metrics (Section 0.5.5)

**Dependencies**:
- Stage 6: `hook_kmeans_analysis.json` (cluster assignments)
- Stage 7: `winning_formulas.json` (winning cluster ID)
- Stage 2: `selected_videos.json` (video metadata)
- Stage 1: `selection_manifest.json` (bucket-scoped top performers)

---

### Section 0.6: Report-Specific Inline Calculations

**Purpose**: Document calculation logic that is specific to individual report templates and doesn't require reusable functions. These calculations should be implemented directly in the corresponding extraction script.

**Organization**: Organized by report template for easy developer reference.

**Key Distinction**:
- **Section 0.5 (Functions)**: Reusable functions used across multiple reports
- **Section 0.6 (Inline Calculations)**: Single-use calculations specific to one report

**Developer Workflow**:
1. Read report field definitions in `Stage8MVP_Reports.md`
2. Check Section 0.5 for reusable functions (if field references a function)
3. Check Section 0.6 for report-specific inline calculations (if field is "calculated")
4. Implement in corresponding extraction script

---

#### 0.6.1: Report 1 (Hashtag → Client) Inline Calculations

**Extraction Script**: `extract_client_data.py`

**Report**: Hashtag → Client (Executive Report)

**Total**: 7 inline calculations covering 8 fields

**Dependencies**: This section uses functions documented in Section 0.5:
- `calculate_avg_views_per_bucket()` (0.5.6.1)
- `calculate_engagement_metrics()` (0.5.5)

---

##### Calculation 1: Array Length Summation

**Fields Using This**:
- Top Performers Count (Report 1, Header Section, Line 77)
- Bottom Performers Count (Report 1, Header Section, Line 78)

**Purpose**: Count total videos selected across all winning buckets for display in report header

**Input Data**:
```json
// {analysis_path}/selection_manifest.json
{
  "videos_by_bucket": {
    "18-33s": {
      "top_performers": ["7540717...", "7539...", ...],    // 33 videos
      "bottom_performers": ["7522...", "7521...", ...]     // 9 videos
    },
    "13-18s": {
      "top_performers": ["7545...", ...],                  // 28 videos
      "bottom_performers": ["7520...", ...]                // 7 videos
    },
    "60-90s": {
      "top_performers": ["7548...", ...],                  // 27 videos
      "bottom_performers": ["7519...", ...]                // 7 videos
    }
  }
}
```

**Implementation**:
```python
# extract_client_data.py - Report 1 Header Section
def calculate_performer_counts(analysis_path):
    """
    Sum array lengths across all buckets in selection manifest.

    Args:
        analysis_path: Path to analysis directory (e.g., .../top_contrastive/)

    Returns:
        dict: {
            "top_performers_count": int,
            "bottom_performers_count": int
        }
    """
    import json

    with open(f"{analysis_path}/selection_manifest.json") as f:
        manifest = json.load(f)

    videos_by_bucket = manifest["videos_by_bucket"]

    # Sum all top_performers array lengths
    top_count = sum(
        len(bucket_data["top_performers"])
        for bucket_data in videos_by_bucket.values()
    )

    # Sum all bottom_performers array lengths
    bottom_count = sum(
        len(bucket_data["bottom_performers"])
        for bucket_data in videos_by_bucket.values()
    )

    return {
        "top_performers_count": top_count,      # Example: 88
        "bottom_performers_count": bottom_count  # Example: 23
    }
```

**Output**:
```python
{
    "top_performers_count": 88,
    "bottom_performers_count": 23
}
```

**Complexity**: Simple (5-10 lines)

**Data Source**: `selection_manifest.json → videos_by_bucket`

---

##### Calculation 2: Bucket Distribution Percentages

**Field Using This**: % per bucket (all 8 rows) (Report 1, Duration Distribution, Line 147)

**Purpose**: Calculate what percentage of scraped videos fall in each duration bucket to show market distribution

**Input Data**:
```json
// {analysis_path}/winner_analysis.json
{
  "bucket_distribution": {
    "0-3s": 146,
    "3-9s": 219,
    "9-13s": 274,
    "13-18s": 402,
    "18-33s": 511,
    "33-60s": 219,
    "60-90s": 37,
    "90-120s": 18
  }
}
```

**Implementation**:
```python
# extract_client_data.py - Report 1 Duration Distribution Section
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
    # Example: {
    #   "0-3s": 8,
    #   "3-9s": 12,
    #   "9-13s": 15,
    #   "13-18s": 22,
    #   "18-33s": 28,
    #   "33-60s": 12,
    #   "60-90s": 2,
    #   "90-120s": 1
    # }
```

**Output**: Dict[str, int] mapping bucket name to percentage

**Complexity**: Simple (8-12 lines)

**Data Source**: `winner_analysis.json → bucket_distribution`

---

##### Calculation 3: Key Insight Percentage

**Field Using This**: Key Insight % (Report 1, Duration Distribution, Line 148)

**Purpose**: Calculate combined percentage for highest-volume buckets to create market insight statement

**Input Data**: Uses output from Calculation 2 (bucket_percentages)

**Implementation**:
```python
# extract_client_data.py - Report 1 Duration Distribution Section
def calculate_key_insight_percentage(bucket_percentages, top_buckets=["13-18s", "18-33s"]):
    """
    Sum percentages for key duration ranges to create insight.

    Args:
        bucket_percentages: Output from calculate_bucket_distribution_percentages()
        top_buckets: List of bucket names to sum (default: 13-18s and 18-33s)

    Returns:
        int: Combined percentage
    """
    key_insight_pct = sum(bucket_percentages[bucket] for bucket in top_buckets)

    return key_insight_pct
    # Example: 22 + 28 = 50
    # Used in report as: "50% of #nutrition content is 13-33s"
```

**Output**: Integer (percentage)

**Complexity**: Simple (1-3 lines)

**Dependencies**: Calculation 2

**Note**: The example hardcodes 13-18s and 18-33s as typical high-volume buckets. In production, you might dynamically identify the top 2 consecutive buckets from the distribution.

---

##### Calculation 4: Star Rating Assignment

**Field Using This**: Star ratings (3 rows) (Report 1, Performance by Duration, Line 174)

**Purpose**: Rank winning buckets by performance metrics and assign 5/4/3 star visual ratings

**Ranking Criteria**:
1. Primary: Average engagement rate (higher is better)
2. Secondary: Average views (higher is better)

**Input Data**: Requires calling documented functions from Section 0.5

**Implementation**:
```python
# extract_client_data.py - Report 1 Performance by Duration Section
def assign_star_ratings(analysis_path, winning_buckets):
    """
    Sort winning buckets by performance and assign star ratings.

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

        # Calculate avg views using documented function (Section 0.5.6.1)
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
            video_id = video["id"]

            # Load metadata from unified_analysis
            unified_path = f"{bucket_path}/analysis/insights/{video_id}_temporal_windows_updated.json"
            with open(unified_path) as f:
                video_data = json.load(f)

            # Use documented function (Section 0.5.5)
            # Note: calculate_engagement_metrics() expects metadata with playCount, diggCount, etc.
            # The metadata is in the video object from selected_videos.json
            engagement = calculate_engagement_metrics({
                "views": video["playCount"],
                "likes": video["diggCount"],
                "comments": video["commentCount"],
                "shares": video["shareCount"],
                "saves": video["collectCount"]
            })
            engagement_rates.append(engagement["engagement_rate"])

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

**Output**:
```python
{
    "star_ratings": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐"],
    "sorted_buckets": [
        {"bucket": "18-33s", "avg_views": 1900000, "avg_engagement": 1.4},
        {"bucket": "13-18s", "avg_views": 2100000, "avg_engagement": 1.2},
        {"bucket": "60-90s", "avg_views": 980000, "avg_engagement": 1.3}
    ]
}
```

**Complexity**: Medium (30-40 lines)

**Dependencies**:
- `calculate_avg_views_per_bucket()` (Section 0.5.6.1)
- `calculate_engagement_metrics()` (Section 0.5.5)

**Data Sources**:
- `selected_videos.json` (video IDs and metadata)
- `winner_analysis.json` (winning buckets)

---

##### Calculation 5: Top Bucket Label

**Field Using This**: Top bucket label (Report 1, Performance by Duration, Line 175)

**Purpose**: Add "← BEST" visual label to highest-performing bucket

**Input Data**: Uses `sorted_buckets` from Calculation 4

**Implementation**:
```python
# extract_client_data.py - Report 1 Performance by Duration Section
def assign_best_labels(sorted_buckets):
    """
    Assign "← BEST" label to top-ranked bucket, empty strings to others.

    Args:
        sorted_buckets: Output from assign_star_ratings()["sorted_buckets"]

    Returns:
        list: ["← BEST", "", ""]
    """
    best_labels = [
        "← BEST" if i == 0 else ""
        for i in range(len(sorted_buckets))
    ]

    return best_labels
    # Example: ["← BEST", "", ""]
```

**Output**: Array[String] with 3 label strings

**Complexity**: Simple (3-5 lines)

**Dependencies**: Calculation 4 (sorted_buckets)

---

##### Calculation 6: Coverage Percentage

**Field Using This**: Coverage percentage (Report 1, Performance by Duration, Line 176)

**Purpose**: Calculate what percentage of top 100 videos fall in the 3 winning buckets

**Input Data**:
```json
// {analysis_path}/winner_analysis.json
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
  }
}
```

**Implementation**:
```python
# extract_client_data.py - Report 1 Performance by Duration Section
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
    # Example: 75.9
    # Used in report as: "These 3 durations represent 75.9% of top-performing content"
```

**Output**: Float (percentage with 1 decimal place)

**Complexity**: Simple (8-12 lines)

**Data Source**: `winner_analysis.json → top_3_buckets, top_100_distribution`

---

##### Calculation 7: Performance Label Assignment

**Field Using This**: Performance labels (Report 1, Creator Profile Priorities, Line 200)

**Purpose**: Map bucket rankings to descriptive labels for creator hiring guidance

**Label Meanings**:
- Rank 1 (5 stars): "highest performance"
- Rank 2 (4 stars): "strong performance + volume"
- Rank 3 (3 stars): "proven success"

**Input Data**: Uses `sorted_buckets` from Calculation 4

**Implementation**:
```python
# extract_client_data.py - Report 1 Creator Profile Priorities Section
def assign_performance_labels(sorted_buckets):
    """
    Map bucket rankings to descriptive performance labels.

    Args:
        sorted_buckets: Output from assign_star_ratings()["sorted_buckets"]

    Returns:
        list: Performance label strings for each bucket
    """
    label_map = {
        0: "highest performance",
        1: "strong performance + volume",
        2: "proven success"
    }

    performance_labels = [
        label_map[i] for i in range(len(sorted_buckets))
    ]

    return performance_labels
    # Example: ["highest performance", "strong performance + volume", "proven success"]
```

**Output**: Array[String] with 3 descriptive labels

**Complexity**: Simple (8-10 lines)

**Dependencies**: Calculation 4 (sorted_buckets)

**Usage in Report**: Labels are displayed alongside bucket names in Creator Profile Priorities:
```
• 18-33s Creators (highest performance: 1.9M avg views)
• 13-18s Creators (strong performance + volume: 2.1M avg views)
• 60-90s Creators (proven success: 980K avg views)
```

---

**Report 1 Calculation Summary**:

| Calculation | Fields | Complexity | Lines of Code | Dependencies |
|-------------|--------|-----------|---------------|--------------|
| 1. Array Length Summation | 2 | Simple | ~10 | None |
| 2. Bucket Distribution % | 1 | Simple | ~12 | None |
| 3. Key Insight % | 1 | Simple | ~3 | Calc 2 |
| 4. Star Rating Assignment | 1 | Medium | ~40 | Functions 0.5.6.1, 0.5.5 |
| 5. Top Bucket Label | 1 | Simple | ~5 | Calc 4 |
| 6. Coverage Percentage | 1 | Simple | ~12 | None |
| 7. Performance Labels | 1 | Simple | ~10 | Calc 4 |
| **Total** | **8 fields** | | **~92 lines** | |

**Implementation Notes**:
- All calculations should be implemented in `extract_client_data.py`
- Calculation 4 is the most complex and should be implemented first (other calculations depend on it)
- Test with actual data from `/data/clients/test_hashtag/` or `/data/clients/test_final/`

---

#### 0.6.2: Report 2 (Hashtag → Creator) Inline Calculations

**Extraction Script**: `extract_creator_data.py`

**Report**: Hashtag → Creator (Content Creator Report)

**Total**: 8 inline calculations covering 19 field instances

**Dependencies**: This section uses functions documented in Section 0.5:
- `calculate_proof_metrics_bucket_scoped()` (0.5.8)
- `aggregate_content_classifications()` (0.5.1)
- `get_top_n_from_field()` (0.5.1.1)

---

##### Calculation 1: Performance Multipliers (The Proof Section)

**Fields Using This**:
- View multiplier (Report 2, The Proof, Line 611)
- Engagement multiplier (Report 2, The Proof, Line 612)

**Purpose**: Calculate how much better top cluster performs vs bottom cluster using ratio format

**Input Data**: Uses output from `calculate_proof_metrics_bucket_scoped()` (Section 0.5.8)

**Implementation**:
```python
# extract_creator_data.py - Report 2 The Proof Section
def calculate_performance_multipliers(proof_metrics):
    """
    Calculate view and engagement multipliers for The Proof section.

    Args:
        proof_metrics: Output from calculate_proof_metrics_bucket_scoped()
            {
                "top_cluster": {"avg_views": 620000, "avg_engagement": 1.2},
                "bottom_cluster": {"avg_views": 380000, "avg_engagement": 0.8}
            }

    Returns:
        dict: {
            "view_multiplier": "1.6x",
            "engagement_multiplier": "1.5x"
        }
    """
    top = proof_metrics["top_cluster"]
    bottom = proof_metrics["bottom_cluster"]

    # Calculate multipliers
    view_multiplier = top["avg_views"] / bottom["avg_views"]
    engagement_multiplier = top["avg_engagement"] / bottom["avg_engagement"]

    # Format as ratio with 1 decimal + 'x' suffix
    return {
        "view_multiplier": f"{view_multiplier:.1f}x",
        "engagement_multiplier": f"{engagement_multiplier:.1f}x"
    }
    # Example: {"view_multiplier": "1.6x", "engagement_multiplier": "1.5x"}
```

**Output**:
```python
{
    "view_multiplier": "1.6x",      # Format: "{ratio:.1f}x"
    "engagement_multiplier": "1.5x"
}
```

**Complexity**: Simple (5 lines)

---

##### Calculation 2: Performance Percentage Increases (The Proof Section)

**Fields Using This**:
- View percentage increase (Report 2, The Proof, Line 613)
- Engagement percentage increase (Report 2, The Proof, Line 614)

**Purpose**: Calculate percentage improvement from bottom to top cluster for marketing impact

**Input Data**: Uses output from `calculate_proof_metrics_bucket_scoped()` (Section 0.5.8)

**Implementation**:
```python
# extract_creator_data.py - Report 2 The Proof Section
def calculate_percentage_increases(proof_metrics):
    """
    Calculate percentage increases for The Proof section.

    Args:
        proof_metrics: Output from calculate_proof_metrics_bucket_scoped()

    Returns:
        dict: {
            "view_pct_increase": 63,
            "engagement_pct_increase": 50
        }
    """
    top = proof_metrics["top_cluster"]
    bottom = proof_metrics["bottom_cluster"]

    # Calculate percentage increases: ((top - bottom) / bottom) × 100%
    view_pct = ((top["avg_views"] - bottom["avg_views"]) / bottom["avg_views"]) * 100
    engagement_pct = ((top["avg_engagement"] - bottom["avg_engagement"]) / bottom["avg_engagement"]) * 100

    # Round to nearest integer
    return {
        "view_pct_increase": round(view_pct),
        "engagement_pct_increase": round(engagement_pct)
    }
    # Example: {"view_pct_increase": 63, "engagement_pct_increase": 50}
```

**Output**:
```python
{
    "view_pct_increase": 63,          # Integer percentage
    "engagement_pct_increase": 50
}
```

**Complexity**: Simple (5 lines)

---

##### Calculation 3: Timing Ranges (Pattern Summary Section)

**Field Using This**: Timing ranges (all 3 steps) (Report 2, Pattern Summary, Line 666)

**Purpose**: Calculate second-by-second timing ranges for Hook/Middle/Closing based on bucket duration

**Input Data**: Bucket name (e.g., "18-33s")

**Implementation**:
```python
# extract_creator_data.py - Report 2 Pattern Summary Section
def calculate_timing_ranges(bucket_name):
    """
    Calculate timing ranges for 3-step pattern based on bucket duration.

    Hook: Always 0-3s (fixed)
    Middle: 3s to (duration - 3s)
    Closing: Last 3s

    Args:
        bucket_name: Duration bucket (e.g., "18-33s", "13-18s")

    Returns:
        list: [hook_range, middle_range, closing_range]
    """
    # Parse bucket duration (use lower bound for calculation)
    # Example: "18-33s" → 18 seconds
    lower_bound = int(bucket_name.split("-")[0].replace("s", ""))

    # Hook: Always 0-3s
    hook_range = "0-3s"

    # Middle: 3s to (duration - 3s)
    middle_end = lower_bound - 3
    middle_range = f"3-{middle_end}s"

    # Closing: Last 3s
    closing_start = lower_bound - 3
    closing_range = f"{closing_start}-{lower_bound}s"

    return [hook_range, middle_range, closing_range]
    # Example for "18-33s": ["0-3s", "3-15s", "15-18s"]
```

**Example Outputs**:
```python
calculate_timing_ranges("18-33s")  # → ["0-3s", "3-15s", "15-18s"]
calculate_timing_ranges("13-18s")  # → ["0-3s", "3-10s", "10-13s"]
calculate_timing_ranges("60-90s")  # → ["0-3s", "3-57s", "57-60s"]
```

**Complexity**: Simple (8 lines)

**Note**: Uses lower bound of bucket range for consistency (18s from "18-33s")

---

##### Calculation 4: snake_case to Title Case Formatter

**Field Using This**: Engagement Drivers (Top 3) (Report 2, Video Category Selection, Line 704)

**Purpose**: Convert snake_case category names to human-readable Title Case for display

**Input Data**: Array of snake_case strings from `get_top_n_from_field()`

**Implementation**:
```python
# extract_creator_data.py - Report 2 Video Category Selection
def format_to_title_case(snake_case_items):
    """
    Convert snake_case to Title Case for display.

    Args:
        snake_case_items: List of snake_case strings

    Returns:
        list: Title Case strings
    """
    return [
        item.replace("_", " ").title()
        for item in snake_case_items
    ]
    # Example: ["before_after_reveal"] → ["Before After Reveal"]
```

**Example Usage**:
```python
# Get snake_case engagement drivers from function
drivers = get_top_n_from_field(bucket_path, "engagement_drivers", n=3, "top")
# Returns: ["personal_testimony", "before_after_reveal", "product_demonstration"]

# Format for display
display_drivers = format_to_title_case(drivers)
# Returns: ["Personal Testimony", "Before After Reveal", "Product Demonstration"]

# Note: For better formatting, you might want custom mappings:
# "before_after_reveal" → "Before/After Reveal" (with slash)
```

**Advanced Implementation** (with custom mappings):
```python
def format_to_title_case(snake_case_items):
    """Convert snake_case to Title Case with custom mappings."""
    custom_mappings = {
        "before_after_reveal": "Before/After Reveal",
        "direct_to_camera": "Direct-to-Camera"
        # Add more as needed
    }

    formatted = []
    for item in snake_case_items:
        if item in custom_mappings:
            formatted.append(custom_mappings[item])
        else:
            formatted.append(item.replace("_", " ").title())

    return formatted
```

**Output**: Array of Title Case strings

**Complexity**: Simple (2-5 lines)

---

##### Calculation 5: CTA Example Phrase Generator

**Field Using This**: CTA Example Phrase (Report 2, Pattern Execution, Line 809)

**Purpose**: Map CTA type to example phrase for creator guidance

**Input Data**: Most common CTA type from `aggregate_content_classifications()`

**Implementation**:
```python
# extract_creator_data.py - Report 2 Pattern Execution Blueprint
def generate_cta_example(cta_type):
    """
    Generate example CTA phrase based on most common type.

    Args:
        cta_type: Most common CTA from aggregation (e.g., "link_in_bio")

    Returns:
        str: Example phrase
    """
    cta_phrases = {
        "link_in_bio": "Link in bio!",
        "save_post": "Save this for later!",
        "comment": "Comment your thoughts!",
        "follow": "Follow for more!",
        "share": "Share with a friend!",
        "tag_friend": "Tag someone who needs this!",
        "none": "Watch until the end!"
    }

    return cta_phrases.get(cta_type, "Link in bio!")
    # Default to "Link in bio!" if type not found
```

**Example Usage**:
```python
# Get most common CTA type
aggregated = aggregate_content_classifications(bucket_path, "top")
cta_counter = aggregated["caption_cta_type"]
most_common_cta = cta_counter.most_common(1)[0][0]  # e.g., "link_in_bio"

# Generate example phrase
example = generate_cta_example(most_common_cta)
# Returns: "Link in bio!"
```

**Output**: String (example phrase)

**Complexity**: Simple (dictionary lookup)

---

##### Calculation 5: Calculate original content percentage

**Purpose**: Calculate percentage of original content (inverse of repost rate)

**Used in**: Report 3 → Page 3 → Section 5 (Content Sourcing Strategy)
- **Field**: Original content %

**Input**:
- `repost_rate` (float): From `extract_mention_analysis()` → `repost_rate`

**Calculation**:
```python
def calculate_original_content_percentage(repost_rate: float) -> int:
    """
    Calculate original content percentage.

    Args:
        repost_rate: Percentage of reposted/affiliate content (0-100)

    Returns:
        Integer percentage of original content (0-100)
    """
    return 100 - int(repost_rate)
```

**Example**:
```python
>>> repost_rate = 42.0
>>> calculate_original_content_percentage(repost_rate)
58  # 100 - 42 = 58% original content

>>> repost_rate = 15.5
>>> calculate_original_content_percentage(repost_rate)
84  # 100 - 15 = 84% original content (rounded down)
```

**Display in Report**:
```
Original Content: 58% (no affiliate mentions)
Reposted/Affiliate Content: 42% (contains @mentions or repost indicators)
```

**Note**: This is a simple inverse calculation. The `repost_rate` from `extract_mention_analysis()` represents videos with @mentions or repost indicators, so `100 - repost_rate` gives the percentage of original content.

---

##### Calculation 6: Hook/CTA Percentage Extraction (Caption Structure Section)

**Fields Using This**: 12 fields (Report 2, Caption Structure, Lines 847-858)
- Hook Type 1-3 + percentages (6 fields)
- CTA Type 1-3 + percentages (6 fields)

**Purpose**: Extract Top 3 items with percentages from Counter objects for caption strategy display

**Input Data**: Uses output from `aggregate_content_classifications()` (Section 0.5.1)

**Implementation**:
```python
# extract_creator_data.py - Report 2 Caption Structure Section
def extract_top_3_with_percentages(counter_obj, total_videos):
    """
    Extract Top 3 items from Counter with percentages.

    Args:
        counter_obj: Counter object from aggregate_content_classifications()
        total_videos: Total video count for percentage calculation

    Returns:
        list: [
            {"name": "question", "percentage": 45},
            {"name": "statement", "percentage": 32},
            {"name": "command", "percentage": 18}
        ]
    """
    top_3 = []

    for item, count in counter_obj.most_common(3):
        percentage = round((count / total_videos) * 100)
        top_3.append({
            "name": item,
            "percentage": percentage
        })

    return top_3
```

**Example Usage**:
```python
# Get aggregated data
aggregated = aggregate_content_classifications(bucket_path, "top")
total = aggregated["total_videos"]

# Extract hook types with percentages
hook_counter = aggregated["caption_hook_type"]
top_hooks = extract_top_3_with_percentages(hook_counter, total)
# Returns: [
#     {"name": "question", "percentage": 45},
#     {"name": "statement", "percentage": 32},
#     {"name": "command", "percentage": 18}
# ]

# Extract CTA types with percentages
cta_counter = aggregated["caption_cta_type"]
top_ctas = extract_top_3_with_percentages(cta_counter, total)
# Returns: [
#     {"name": "link_in_bio", "percentage": 67},
#     {"name": "save_post", "percentage": 21},
#     {"name": "comment", "percentage": 9}
# ]

# Access individual fields for report
hook_type_1 = top_hooks[0]["name"]        # "question"
hook_pct_1 = top_hooks[0]["percentage"]   # 45
cta_type_1 = top_ctas[0]["name"]          # "link_in_bio"
cta_pct_1 = top_ctas[0]["percentage"]     # 67
```

**Output**:
```python
[
    {"name": str, "percentage": int},
    {"name": str, "percentage": int},
    {"name": str, "percentage": int}
]
```

**Complexity**: Simple (8 lines)

**Note**: This calculation handles 12 separate fields in the report (6 hook + 6 CTA)

---

**Report 2 Calculation Summary**:

| Calculation | Fields | Complexity | Lines of Code | Dependencies |
|-------------|--------|-----------|---------------|--------------|
| 1. Performance Multipliers | 2 | Simple | ~8 | Function 0.5.8 |
| 2. Percentage Increases | 2 | Simple | ~8 | Function 0.5.8 |
| 3. Timing Ranges | 1 | Simple | ~12 | None |
| 4. snake_case to Title Case | 1 | Simple | ~5 | None |
| 5. CTA Example Generator | 1 | Simple | ~12 (dictionary) | Function 0.5.1 |
| 6. Top 3 with Percentages | 12 | Simple | ~10 | Function 0.5.1 |
| **Total** | **19 field instances** | | **~55 lines** | |

**Implementation Notes**:
- All calculations should be implemented in `extract_creator_data.py`
- Calculations 1-2 depend on `calculate_proof_metrics_bucket_scoped()` output
- Calculation 6 handles 12 fields but uses same function for both hook and CTA types
- Test with actual data from `/data/clients/test_final/hashtags/`

---

#### 0.6.3: Report 3 (Single Competitor → Client) Inline Calculations

**Extraction Script**: `extract_competitor_data.py`

**Report**: Single Competitor → Client (Deep Dive Report)

**Total**: 5 inline calculations covering 6 fields

**Dependencies**: This section uses functions documented in Section 0.5:
- `calculate_competitor_bucket_avg_views()` (0.5.6.2)
- `calculate_competitor_bucket_avg_engagement()` (0.5.6.3)
- `calculate_bucket_distribution()` (0.5.6.1)

---

##### Calculation 1: `rank_competitor_top_buckets()`

**Purpose**: Rank a single competitor's top 3 buckets by performance and assign star ratings

**Used in**: Report 3 → Page 2 → Section 2 (Performance by Duration)
- **Field 5**: Star ratings (3 rows)
- **Field 6**: Sweet spot bucket

**Input Parameters**:
- `client_id` (string): Client identifier (e.g., "test_competitor")
- `competitor_handle` (string): Competitor handle with @ symbol (e.g., "@drinkpoppi")

**Process**:
1. Get competitor's `top_3_buckets` from `winner_analysis.json`
2. For each bucket:
   - Get avg views using `calculate_competitor_bucket_avg_views()`
   - Get avg engagement using `calculate_competitor_bucket_avg_engagement()`
   - Calculate composite score: `(normalized_views * 100) + engagement`
3. Sort buckets by composite score (DESC)
4. Assign star ratings: Rank 1 = ⭐⭐⭐⭐⭐, Rank 2 = ⭐⭐⭐⭐, Rank 3 = ⭐⭐⭐⭐

**Returns**: List of dicts with bucket rankings

**Example Output**:
```python
[
    {
        "bucket": "18-33s",
        "rank": 1,
        "avg_views": 620000,
        "avg_engagement": 1.5,
        "composite_score": 101.5,
        "stars": "⭐⭐⭐⭐⭐",
        "is_sweet_spot": True
    },
    {
        "bucket": "13-18s",
        "rank": 2,
        "avg_views": 580000,
        "avg_engagement": 1.3,
        "composite_score": 95.0,
        "stars": "⭐⭐⭐⭐",
        "is_sweet_spot": False
    },
    {
        "bucket": "33-60s",
        "rank": 3,
        "avg_views": 490000,
        "avg_engagement": 1.4,
        "composite_score": 81.4,
        "stars": "⭐⭐⭐⭐",
        "is_sweet_spot": False
    }
]
```

**Implementation**:
```python
def rank_competitor_top_buckets(client_id: str, competitor_handle: str) -> list[dict]:
    """
    Rank competitor's top 3 buckets by performance.

    Args:
        client_id: Client identifier
        competitor_handle: Competitor handle with @ symbol

    Returns:
        List of dicts with rankings, sorted by performance DESC
    """
    import json
    import os

    # Discover analysis directory
    competitor_dir = competitor_handle.lstrip('@')
    base_path = f"/data/clients/{client_id}/competitors/{competitor_dir}"
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(f"No analysis directory found for {competitor_handle}")

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load top 3 buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    top_3_buckets = winner_data["top_3_buckets"]

    # Collect performance data for each bucket
    bucket_data = []
    for bucket in top_3_buckets:
        avg_views = calculate_competitor_bucket_avg_views(client_id, competitor_handle, bucket)
        avg_engagement = calculate_competitor_bucket_avg_engagement(client_id, competitor_handle, bucket)

        bucket_data.append({
            "bucket": bucket,
            "avg_views": avg_views,
            "avg_engagement": avg_engagement
        })

    # Normalize views and calculate composite scores
    max_views = max(b["avg_views"] for b in bucket_data)

    for bucket in bucket_data:
        normalized_views = (bucket["avg_views"] / max_views) * 100
        composite_score = normalized_views + bucket["avg_engagement"]
        bucket["composite_score"] = composite_score

    # Sort by composite score (DESC)
    bucket_data.sort(key=lambda b: b["composite_score"], reverse=True)

    # Assign ranks and star ratings
    star_map = {1: "⭐⭐⭐⭐⭐", 2: "⭐⭐⭐⭐", 3: "⭐⭐⭐⭐"}

    for idx, bucket in enumerate(bucket_data, start=1):
        bucket["rank"] = idx
        bucket["stars"] = star_map[idx]
        bucket["is_sweet_spot"] = (idx == 1)

    return bucket_data
```

**Report Usage**:
- **Star ratings (Field 5)**: Use `stars` field for each bucket
- **Sweet spot (Field 6)**: Get bucket where `is_sweet_spot == True`

---

##### Calculation 2: `calculate_top_3_coverage()`

**Purpose**: Calculate what % of competitor's content is in their top 3 performing buckets

**Used in**: Report 3 → Page 2 → Section 2 (Performance by Duration)
- **Field 7**: Coverage percentage

**Input**:
- `bucket_percentages` (dict): Output from `calculate_bucket_distribution()`
- `top_3_buckets` (list): From `winner_analysis.json` → `top_3_buckets`

**Calculation**:
```python
def calculate_top_3_coverage(bucket_percentages: dict, top_3_buckets: list[str]) -> int:
    """
    Sum percentages of top 3 buckets.

    Args:
        bucket_percentages: Dict from calculate_bucket_distribution()
        top_3_buckets: List of 3 bucket names

    Returns:
        Integer percentage (e.g., 72)
    """
    return sum(bucket_percentages[bucket] for bucket in top_3_buckets)
```

**Example**:
```python
>>> bucket_pct = {"0-3s": 8, "3-9s": 36, "9-13s": 18, "18-33s": 15, ...}
>>> top_3 = ["3-9s", "9-13s", "18-33s"]
>>> calculate_top_3_coverage(bucket_pct, top_3)
69  # 36 + 18 + 15 = 69%
```

**Note**: This is a simple helper calculation (one-liner). Could be done inline without a dedicated function, but documented here for completeness and developer clarity.

---

##### Calculation 3: Format engagement driver descriptions

**Purpose**: Convert engagement driver names from snake_case to human-readable descriptions

**Used in**: Report 3 → Page 3 → Section 1 (Content DNA)
- **Field 5**: Engagement driver descriptions

**Input**:
- `engagement_drivers` (list): List of engagement driver names in snake_case from Stage 2.7
  - Example: `["before_after_reveal", "specific_metrics", "personal_testimony", "expert_credentials"]`

**Process**:
1. Convert snake_case to Title Case: `before_after_reveal` → `Before After Reveal`
2. Add contextual description based on known patterns

**Mapping Table**:
```python
ENGAGEMENT_DRIVER_DESCRIPTIONS = {
    "before_after_reveal": "Visual transformations",
    "specific_metrics": '"Lost 15 lbs in 30 days"',
    "personal_testimony": '"This worked for me..."',
    "expert_credentials": '"Registered nutritionist here..."',
    "product_demonstration": "Showing product in use",
    "social_proof": "User reviews and testimonials",
    "urgency_scarcity": "Limited time offers",
    "emotional_appeal": "Tugging at heartstrings",
    # Add more as Stage 2.7 taxonomy expands
}
```

**Implementation**:
```python
def format_engagement_driver_description(driver: str) -> str:
    """
    Get human-readable description for engagement driver.

    Args:
        driver: Snake_case driver name (e.g., "before_after_reveal")

    Returns:
        Human-readable description (e.g., "Visual transformations")
    """
    # Fallback: Convert to title case if not in mapping
    return ENGAGEMENT_DRIVER_DESCRIPTIONS.get(
        driver,
        driver.replace("_", " ").title()
    )
```

**Example**:
```python
>>> drivers = ["before_after_reveal", "specific_metrics", "personal_testimony"]
>>> [format_engagement_driver_description(d) for d in drivers]
["Visual transformations", '"Lost 15 lbs in 30 days"', '"This worked for me..."']
```

**Note**: Descriptions should be concise (2-6 words) and provide context about what the driver means in practice.

---

##### Calculation 4: Determine hashtag strategy type

**Purpose**: Classify competitor's hashtag strategy as "Diversified" or "Focused" based on unique hashtag count

**Used in**: Report 3 → Page 3 → Section 3 (Hashtag Strategy)
- **Field**: Strategy type

**Input**:
- `total_unique_hashtags` (int): From `extract_hashtag_analysis()` → `total_unique_hashtags`

**Logic**:
```python
def determine_hashtag_strategy_type(total_unique_hashtags: int) -> str:
    """
    Classify hashtag strategy based on diversity.

    Args:
        total_unique_hashtags: Count of distinct hashtags used

    Returns:
        "Diversified" if > 20, else "Focused"
    """
    return "Diversified" if total_unique_hashtags > 20 else "Focused"
```

**Threshold Rationale**:
- **≤ 20 hashtags** = "Focused" strategy
  - Competitor concentrates on core hashtags
  - Consistent messaging across content
  - Example: Brand focuses on 10-15 key hashtags

- **> 20 hashtags** = "Diversified" strategy
  - Competitor uses varied hashtags across content
  - Broader topic coverage
  - Example: Brand uses 30+ hashtags to reach different audiences

**Example**:
```python
>>> total = 28
>>> determine_hashtag_strategy_type(total)
"Diversified"  # 28 > 20

>>> total = 15
>>> determine_hashtag_strategy_type(total)
"Focused"  # 15 <= 20
```

**Display in Report**:
```
Strategy Type: Diversified (28 hashtags across content)
```

**Note**: This is a simple threshold-based classification. The threshold of 20 is based on typical TikTok hashtag usage patterns where most accounts use 10-15 core hashtags, while diversified strategies employ 25-40+ hashtags.

---

#### 0.6.4: Report 4 (Multi-Competitor → Client) Inline Calculations

**Extraction Script**: `extract_multi_competitor_data.py`

**Status**: To be documented

---

### Section 3: Data Extraction Scripts (3.25 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 3.1 | Build `extract_creator_data.py` + QR generation | Developer | 1.25 days | Report 2: Hashtag → Creator (3 formulas + 6 QR codes) |
| 3.2 | Build `extract_client_data.py` | Developer | 1 day | Report 1: Hashtag → Client (no QR codes) |
| 3.3 | Build `extract_competitor_data.py` + QR generation | Developer | 0.5 days | Report 3: Single Competitor → Client (1 QR code) |
| 3.4 | Build `extract_multi_competitor_data.py` + QR generation | Developer | 0.5 days | Report 4: Multi-Competitor → Client (1 QR code per competitor) |

**Total Effort**: 3.25 days (unchanged - Task 3.3 split into two 0.5-day tasks)

**QR Code Generation Summary**:
- **Report 1**: No QR codes (executive dashboard only)
- **Report 2**: 6 QR codes per hashtag (2 per formula: top/bottom performer × 3 formulas)
- **Report 3**: 1 QR code per competitor (top performer video example)
- **Report 4**: 1 QR code per competitor (varies: 2-5 competitors = 2-5 QR codes)
- **Implementation**: Python `qrcode` library (free, BSD license)
- **Tracking**: Direct TikTok URLs (no tracking service, simplest for MVP)
- **Format**: PNG images, ~5KB each, 1" × 1" size for PDF templates

**Script Requirements**:
- **Input**: JSON files from Stages 1, 6, 7 (existing RumiAI ML pipeline outputs)
- **Output**: ✅ **Excel files (.xlsx)** with clearly labeled sections and data fields
- **Excel Library**: Use `pandas` + `openpyxl` for Excel generation
- **Sheet Structure**: Single tab per report, two-column format (Field Name | Value)
- **Error handling**: Clear error messages if JSON missing/malformed
- **CLI interface**: Simple command-line invocation

**Data Validation** (Decision: Alternative A - Minimal Validation):
- **Success/Failure only**: Scripts print completion status and output paths
- **No automatic checks**: No validation of data quality, ranges, or completeness
- **Manual review**: User opens Excel to verify accuracy (10-15 min per report)
- **Rationale**: Simplest for MVP, avoids complexity, user review step already planned in workflow

**Console Output Pattern**:
```bash
$ python extract_creator_data.py --client acme --hashtag nutrition

Running extraction for hashtag: #nutrition
Processing 3 winning buckets...
Generating 6 QR codes...

✓ Extraction complete
  Excel: /data/clients/acme/hashtags/nutrition/top_contrastive/nutrition_creator_data.xlsx
  QR codes: 6 generated in qr_codes/
```

**Error Handling** (will cause script to exit with error):
- Missing JSON files (e.g., `winner_analysis.json` not found)
- Malformed JSON (cannot parse)
- File write permissions (cannot create Excel or QR codes)
- Missing required fields in JSON (e.g., `top_3_buckets` array empty)

---

#### 3.0: CLI Design & Script Architecture

**Decision**: ✅ **Script-Based Architecture (Alternative A)**

**Rationale**:
1. **Clarity for MVP**: Explicit script names make it immediately clear which report type you're generating, reducing errors during manual onboarding workflow
2. **Matches mental model**: 4 distinct report types (different audiences, structures) → 4 distinct scripts
3. **Code reuse via imports**: Shared functions live in Section 0.5/0.6, imported by all scripts
4. **Future-proof**: New report types = new scripts, without touching existing ones
5. **Documentation clarity**: Onboarding workflow tables can reference named scripts

**Architecture**:

```
Stage 8 Extraction Scripts (NEW - Post-processing layer)
├── extract_client_data.py          → Report 1: Hashtag → Client
├── extract_creator_data.py         → Report 2: Hashtag → Creator
├── extract_competitor_data.py      → Report 3: Single Competitor
├── extract_multi_competitor_data.py → Report 4: Multi-Competitor
└── report_utils.py                 → Shared functions from Section 0.5/0.6

RumiAI ML Pipeline (EXISTING - Unchanged)
└── rumiai_runner.py / rumiai_ml_batch.py → Stages 1-7 analysis
```

**Important**: Stage 8 extraction scripts are **separate** from the RumiAI ML pipeline CLI. They consume RumiAI's JSON outputs but do not modify the ML pipeline architecture.

---

**Script Mapping Table**:

| Script Name | Report Type | Template Ref | CLI Example | Output |
|-------------|-------------|--------------|-------------|--------|
| `extract_client_data.py` | Report 1: Hashtag → Client | Stage8MVP_Reports.md §1 | `python extract_client_data.py --client acme --hashtag nutrition` | Excel file (1 tab: all pages) |
| `extract_creator_data.py` | Report 2: Hashtag → Creator | Stage8MVP_Reports.md §2 | `python extract_creator_data.py --client acme --hashtag nutrition` | Excel file (3 tabs: 1 per formula) + 6 QR codes |
| `extract_competitor_data.py` | Report 3: Single Competitor | Stage8MVP_Reports.md §3 | `python extract_competitor_data.py --client acme --competitor drinkpoppi` | Excel file (1 tab: all pages) |
| `extract_multi_competitor_data.py` | Report 4: Multi-Competitor | Stage8MVP_Reports.md §4 | `python extract_multi_competitor_data.py --client acme --competitors drinkpoppi,vitalproteins,nike` | Excel file (1 tab: all pages) |

---

**Shared Functions Architecture**:

All scripts import from `report_utils.py` module containing Section 0.5/0.6 functions:

**Section 0.5 Functions** (Qualitative Data Processing):
- `aggregate_content_classifications()` - Aggregate Stage 2.7 classifications
- `get_top_n_from_field()` - Extract Top N items from aggregated data
- `select_qr_code_videos()` - Select top/bottom videos for QR codes
- `extract_hashtag_analysis()` - Extract hashtag patterns across buckets
- `extract_mention_analysis()` - Extract @mention patterns for content sourcing
- `calculate_engagement_metrics()` - Calculate real engagement rates from metadata
- `get_visual_direction()` - Categorize visual framing (close-up, medium, wide)
- `calculate_proof_metrics_bucket_scoped()` - Compare cluster performance within bucket

**Section 0.6 Functions** (Inline Calculations):
- Avg views per bucket formatting (K/M suffix)
- Star ratings (5★ to 3★ based on engagement + views)
- Coverage percentage (top 3 buckets as % of total)
- Market leader identification (composite score)
- Bucket distribution percentages
- High-volume bucket markers (>20% threshold)
- And ~20 other inline calculations

**Scripts are thin orchestration layers** that:
1. Parse CLI arguments
2. Build file paths from arguments (e.g., `/data/clients/{client}/hashtags/{hashtag}/...`)
3. Call shared functions in correct order for specific report structure
4. Format output following Stage8MVP_Reports.md template specifications
5. Write to Google Sheets with appropriate tabs/sections
6. Save output URL reference file

---

**Common CLI Parameters**:

**Required for all scripts**:
```bash
--client <client_id>       # Example: acme, rippleos
```

**Hashtag scripts only** (Reports 1 & 2):
```bash
--hashtag <target>         # Example: nutrition, vitamin
--mode <mode>              # Optional, default: top (options: top, recent)
--strategy <strategy>      # Optional, default: contrastive (options: contrastive, engagement)
```

**Competitor scripts only** (Reports 3 & 4):
```bash
# Report 3 (single):
--competitor <handle>      # Example: drinkpoppi (no @ symbol)

# Report 4 (multi):
--competitors <list>       # Example: drinkpoppi,vitalproteins,nike (comma-separated, 2-5 competitors)
```

---

**Workflow Example**:

```bash
# Step 1: Run RumiAI ML Pipeline (existing system - Stages 1-7)
python rumiai_runner.py --client acme --hashtag nutrition --stages 1-7

# Outputs JSON files:
# - /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json
# - /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/selected_videos.json
# - /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/ml_analysis/kmeans_analysis.json
# - /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/ml_analysis/llm/winning_formulas.json
# - etc.

# Step 2: Extract Report Data (new Stage 8 system)
python extract_client_data.py --client acme --hashtag nutrition

# Reads JSON files from Step 1
# Calls Section 0.5/0.6 functions
# Outputs: nutrition_client_data.xlsx
```

---

#### 3.0.1: Output Format & Designer Workflow

**Decision**: ✅ **Excel Files with Single-Tab Structure (Alternative B)**

**Rationale**:
1. **Simplicity for MVP**: No Google API setup, no authentication, no external dependencies
2. **Bug minimization**: Script generates files from scratch (no template dependencies that can break)
3. **Offline workflow**: Designer can work without internet connection
4. **Familiar tooling**: Most designers already have Excel/LibreOffice
5. **Implementation speed**: ~15 lines of code vs ~50 for template-based approach

**Architecture**: Scripts generate Excel files from scratch (no pre-existing template files)

---

**Excel File Structure**:

**Single tab per report** with two-column format:

```
Column A: Field Name (UPPERCASE_WITH_UNDERSCORES)
Column B: Value (extracted data)
```

**Section dividers** for page/section organization:

```
| Field Name                     | Value                    |
|================================|==========================|
| PAGE_1_SCALE_OF_ANALYSIS       |                          |
|================================|==========================|
|                                |                          |
| HASHTAG                        | #nutrition               |
| ANALYSIS_PERIOD                | Past 2-3 months          |
| TOTAL_VIDEOS                   | 1826                     |
| BUCKET_1_NAME                  | 18-33s                   |
| BUCKET_1_PCT                   | 43                       |
| BUCKET_2_NAME                  | 13-18s                   |
| BUCKET_2_PCT                   | 12                       |
| BUCKET_3_NAME                  | 60-90s                   |
| BUCKET_3_PCT                   | 11                       |
| TOP_PERFORMERS_COUNT           | 88                       |
| BOTTOM_PERFORMERS_COUNT        | 23                       |
|                                |                          |
|================================|==========================|
| PAGE_2_HASHTAG_INTELLIGENCE    |                          |
|================================|==========================|
|                                |                          |
| BUCKET_0_3S_PCT                | 8                        |
| BUCKET_3_9S_PCT                | 12                       |
| BUCKET_9_13S_PCT               | 15                       |
| ...                            | ...                      |
```

**Formatting**:
- **Plain text only** (no colors, bold, borders for MVP simplicity)
- **Empty rows** between sections for visual separation
- **Section headers** in Field Name column with equals signs

**Field Naming Convention**: `UPPERCASE_WITH_UNDERSCORES`
- **Why**: Easy to spot as placeholders when designer manually searches in PDF templates
- **Examples**: `HASHTAG`, `TOTAL_VIDEOS`, `BUCKET_1_NAME`, `AVG_VIEWS_BUCKET_1`
- **Multi-item fields**: Use numbered suffixes (e.g., `KEYWORD_1`, `KEYWORD_2`, `KEYWORD_3`)

---

**Multi-Value Field Structure** (Decision: Alternative A - Flat List with Numbered Suffixes):

All fields are single-cell values in Column B. Multi-value fields (lists, arrays) use numbered suffixes.

**Rationale**:
1. **Simplest implementation**: Sequential row generation, no conditional logic for tables
2. **Consistent workflow**: Designer always copies from Column B (never selects multiple cells)
3. **No special cases**: Same two-column pattern throughout entire Excel file
4. **Fewer bugs**: ~20 lines of code vs ~50+ for grouped/table formats

**Examples**:

**Top 3 Keywords** (3 rows):
```
| KEYWORD_1         | #guthealth           |
| KEYWORD_2         | #protein             |
| KEYWORD_3         | #antiinflammatory    |
```

**Top 3 Winning Buckets with Performance Data** (12 rows):
```
| BUCKET_1_NAME     | 18-33s               |
| BUCKET_1_PCT      | 43                   |
| BUCKET_1_AVG_VIEWS| 490K                 |
| BUCKET_1_AVG_ENG  | 1.4                  |
|                   |                      |
| BUCKET_2_NAME     | 13-18s               |
| BUCKET_2_PCT      | 12                   |
| BUCKET_2_AVG_VIEWS| 520K                 |
| BUCKET_2_AVG_ENG  | 1.2                  |
|                   |                      |
| BUCKET_3_NAME     | 60-90s               |
| BUCKET_3_PCT      | 11                   |
| BUCKET_3_AVG_VIEWS| 310K                 |
| BUCKET_3_AVG_ENG  | 1.3                  |
```

**Top 5 Pain Points** (5 rows):
```
| PAIN_POINT_1      | Bloating             |
| PAIN_POINT_2      | Low Energy           |
| PAIN_POINT_3      | Weight Loss          |
| PAIN_POINT_4      | Gut Health           |
| PAIN_POINT_5      | Brain Fog            |
```

**Pattern**:
- Single items: `FIELD_NAME | value`
- Lists of N items: `FIELD_NAME_1 | value1`, `FIELD_NAME_2 | value2`, ..., `FIELD_NAME_N | valueN`
- Complex objects (e.g., buckets with multiple properties): `BUCKET_1_PROPERTY`, `BUCKET_2_PROPERTY`, etc.
- Empty rows between major groups for visual separation (optional)

**Section Header Granularity**:

**Page-level dividers** (required):
```
|================================|==========================|
| PAGE_1_SCALE_OF_ANALYSIS       |                          |
|================================|==========================|
```

**Subsection dividers** (optional, use sparingly ~5-8 per report):
```
| --- Header Section ---         |                          |
| --- Top Keywords ---           |                          |
| --- Winning Buckets ---        |                          |
```

**Guidelines**:
- Use page dividers for every PDF page break
- Use subsection dividers only for major conceptual groups (not every field cluster)
- Keep subsection labels short and descriptive
- Add empty row after subsection divider before first field

---

**Designer Workflow** (Manual Population - No Automation for MVP):

1. **Run extraction script** to generate populated Excel file
   ```bash
   python extract_client_data.py --client acme --hashtag nutrition
   # Output: /data/clients/acme/hashtags/nutrition/top_contrastive/nutrition_client_data.xlsx
   ```

2. **Review data in Excel** (10-15 min)
   - Check for missing values
   - Verify numbers make sense
   - Edit if needed (Excel is editable)

3. **Open PDF template** (InDesign, Illustrator, Figma, etc.)

4. **Manual copy-paste workflow**:
   - Locate field in template (e.g., page shows "{{HASHTAG}} Analysis")
   - Find field in Excel (search for `HASHTAG` in Column A)
   - Copy value from Column B (`#nutrition`)
   - Paste into template, replacing `{{HASHTAG}}`
   - Repeat for all fields

5. **Export PDF** when complete

**Time estimate** (from Stage8MVP.md workflow tables):
- Review Excel: 10-15 min
- Populate template: 20 min (Report 1) to 3 hrs (Report 2 - 9 formulas)
- Export PDF: 2-5 min

---

**Report-Specific Output Files**:

| Report Type | Output File Name | Structure |
|-------------|------------------|-----------|
| Report 1: Hashtag → Client | `{hashtag}_client_data.xlsx` | Single tab: ~50 fields across 3 pages |
| Report 2: Hashtag → Creator | `{hashtag}_creator_data.xlsx` | 3 tabs (1 per winning bucket formula): ~40 fields per tab across 2 pages |
| Report 3: Single Competitor | `{competitor}_analysis_data.xlsx` | Single tab: ~60 fields across 3 pages |
| Report 4: Multi-Competitor | `multi_competitor_{N}comp_data.xlsx` | Single tab: ~80+ fields across 4 pages (varies by N competitors) |

**Note**: Report 2 uses **3 tabs in one Excel file** (one per winning bucket formula), making it easy for designer to navigate between the 3 creator PDF templates.

---

**Implementation Libraries**:

```python
# Required packages
import pandas as pd  # DataFrame manipulation
import openpyxl      # Excel file writing (pandas dependency)

# Basic implementation pattern
data = {
    'Field Name': ['HASHTAG', 'TOTAL_VIDEOS', 'BUCKET_1_NAME', ...],
    'Value': ['#nutrition', 1826, '18-33s', ...]
}
df = pd.DataFrame(data)
df.to_excel('nutrition_client_data.xlsx', sheet_name='Report_Data', index=False)
```

**No external dependencies**: Both `pandas` and `openpyxl` are standard Python data science libraries, already used in RumiAI pipeline.

---

**Why Not Google Sheets?** (Comparison)

| Aspect | Google Sheets | Excel (Chosen) |
|--------|---------------|----------------|
| Setup | Requires Google API credentials, OAuth flow | No setup - just write file |
| Script complexity | ~50 lines (auth + API calls) | ~15 lines (pandas.to_excel) |
| Designer access | Needs Google account, share links | Opens file directly |
| Review/Edit | Online, any device | Desktop app |
| Dependencies | `gspread` or `google-api-python-client` | `pandas` + `openpyxl` (already installed) |
| Bugs | API rate limits, auth token expiry | None |

**Decision**: Excel is simpler, more reliable, and faster to implement for MVP.

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

**Excel Structure**:
- **File Name**: `nutrition_creator_data.xlsx`
- **3 tabs**: One per winning bucket formula (Formula_18-33s, Formula_13-18s, Formula_60-90s)
- **Each tab structure**: Two-column format (Field Name | Value) with ~40 fields across 2 pages

**QR Code Generation**:
- **6 QR codes total** per hashtag (2 per formula: top + bottom performer)
- **Output directory**: `qr_codes/` subfolder (keeps images separate from Excel)
- **File naming**: `{hashtag}_{bucket}_top.png`, `{hashtag}_{bucket}_bottom.png`
- **Example**: `nutrition_18-33s_top.png`, `nutrition_18-33s_bottom.png`, ...
- **Video selection**: Highest view count from top_performers (top QR) and bottom_performers (bottom QR)
- **URL source**: `selected_videos.json` → `webVideoUrl` field
- **QR encoding**: Direct TikTok URLs (no tracking, simplest for MVP)
- **Library**: Python `qrcode[pil]` (free, BSD license)

**Output Location**:
```
/data/clients/acme/hashtags/nutrition/top_contrastive/
├── nutrition_creator_data.xlsx
└── qr_codes/
    ├── nutrition_18-33s_top.png
    ├── nutrition_18-33s_bottom.png
    ├── nutrition_13-18s_top.png
    ├── nutrition_13-18s_bottom.png
    ├── nutrition_60-90s_top.png
    └── nutrition_60-90s_bottom.png
```

**Excel includes QR metadata per formula tab**:
```
| QR_CODE_TOP_FILE       | nutrition_18-33s_top.png           |
| QR_CODE_TOP_URL        | https://www.tiktok.com/@user/...   |
| QR_CODE_TOP_VIEWS      | 620K                               |
| QR_CODE_BOTTOM_FILE    | nutrition_18-33s_bottom.png        |
| QR_CODE_BOTTOM_URL     | https://www.tiktok.com/@user/...   |
| QR_CODE_BOTTOM_VIEWS   | 95K                                |
```

**Designer workflow**: Open Excel tab → See QR filenames → Open `qr_codes/` folder → Drag 2 QR images into PDF template

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

**Purpose**: Extract single competitor deep dive analysis data (Report 3: Single Competitor → Client)

**Report Type**: Report 3 from Stage8MVP_Reports.md Section 3

**Engagement Metrics**: Uses `calculate_engagement_metrics()` (Section 0.5.5) to calculate real engagement rates from Apify metadata

**Input**:
- Competitor Stage 1 `winner_analysis.json` (bucket distribution)
- Competitor Stage 2 `selected_videos.json` (video metadata including `views`, `likes`, `comments`, `shares`, `saves`, hashtags, @mentions)
- Competitor Stage 2.7 `content_analysis` (content categories, hook strategies, pain points, keywords)
- Competitor Stage 6 ML analysis JSONs
- Competitor Stage 7 `winning_formulas.json`

**Output Format**:
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

```

**For detailed report template structure and dynamic field mappings, see:**
→ `Stage8MVP_Reports.md` Section "3. Handle/Single Competitor → Client (Deep Dive Report)"

**CLI Usage**:
```bash
python extract_competitor_data.py --client acme --competitor drinkpoppi
```

**Excel Structure**:
- **File Name**: `drinkpoppi_analysis_data.xlsx`
- **1 tab**: Single tab with all pages/sections (two-column format)
- **Structure**: ~60 fields across 3 pages matching Stage8MVP_Reports.md Section 3

**QR Code Generation**:
- **1 QR code** (competitor's top performing video)
- **Output directory**: `qr_codes/` subfolder
- **File naming**: `{competitor}_top.png`
- **Example**: `drinkpoppi_top.png`
- **Video selection**: Highest view count from competitor's winning bucket top_performers
- **URL source**: `selected_videos.json` → `webVideoUrl` field
- **QR encoding**: Direct TikTok URL (no tracking)

**Output Location**:
```
/data/clients/acme/competitors/drinkpoppi/
├── drinkpoppi_analysis_data.xlsx
└── qr_codes/
    └── drinkpoppi_top.png
```

**Excel includes QR metadata**:
```
| QR_CODE_FILE          | drinkpoppi_top.png                 |
| QR_CODE_URL           | https://www.tiktok.com/@user/...   |
| QR_CODE_VIEWS         | 820K                               |
| QR_CODE_ENGAGEMENT    | 1.5                                |
| QR_CODE_DURATION      | 45s                                |
| QR_CODE_BUCKET        | 33-60s                             |
```

---

#### 3.4: `extract_multi_competitor_data.py`

**Purpose**: Extract multi-competitor market intelligence data (Report 4: Multi-Competitor → Client)

**Report Type**: Report 4 from Stage8MVP_Reports.md Section 4

**Scope**: Pure market intelligence (no client comparison) - analyzes 2-5 competitors

**Engagement Metrics**: Uses `calculate_engagement_metrics()` (Section 0.5.5) to calculate real engagement rates from Apify metadata

**Input** (per competitor):
- Competitor Stage 1 `winner_analysis.json` (bucket distribution)
- Competitor Stage 2 `selected_videos.json` (video metadata including `views`, `likes`, `comments`, `shares`, `saves`, hashtags, @mentions)
- Competitor Stage 2.7 `content_analysis` (content categories, hook strategies, pain points, keywords)
- Competitor Stage 6 ML analysis JSONs
- Competitor Stage 7 `winning_formulas.json`

**Output Format**:

Google Sheet with 4 tabs containing data for multi-competitor market intelligence:

**Tab 1: Page_1_Market_Overview**
- Performance Rankings table (all competitors sorted by composite score)
- Market leader identification
- Analysis scope per competitor

**Tab 2: Page_2_Content_Strategy**
- Bucket distribution matrix (8 buckets × N competitors)
- Performance by duration matrix (winning buckets × N competitors)
- Posting frequency comparison
- Market patterns per bucket

**Tab 3: Page_3_Creative_Intelligence**
- Content DNA (top 2 content categories per bucket per competitor)
- Execution Playbook (top 2 hook strategies, CTAs, pain points, keywords per bucket)
- Hashtag strategy comparison
- Caption strategy comparison
- Content sourcing strategy

**Tab 4: Page_4_Visual_Examples**
- QR codes (1 per competitor - top performing video)
- Video stats (views, engagement, duration, formula name)
- Key pattern elements per video

**For detailed report template structure and dynamic field mappings, see:**
→ `Stage8MVP_Reports.md` Section "4. Handle/Multiple Competitor → Client (Market Intelligence Report)"

**CLI Usage**:
```bash
python extract_multi_competitor_data.py --client acme --competitors drinkpoppi,vitalproteins,nike
```

**Excel Structure**:
- **File Name**: `multi_competitor_3comp_data.xlsx`
- **1 tab**: Single tab with all pages/sections (two-column format)
- **Structure**: ~80+ fields across 4 pages (varies by number of competitors)

**QR Code Generation**:
- **1 QR code per competitor** (varies: 2-5 competitors = 2-5 QR codes)
- **Output directory**: `qr_codes/` subfolder
- **File naming**: `{competitor}_top.png`
- **Example**: `drinkpoppi_top.png`, `vitalproteins_top.png`, `nike_top.png`
- **Video selection**: Highest view count from each competitor's winning bucket top_performers
- **URL source**: `selected_videos.json` → `webVideoUrl` field per competitor
- **QR encoding**: Direct TikTok URLs (no tracking)

**Output Location**:
```
/data/clients/acme/competitors/
├── multi_competitor_3comp_data.xlsx
└── qr_codes/
    ├── drinkpoppi_top.png
    ├── vitalproteins_top.png
    └── nike_top.png
```

**Excel includes QR metadata per competitor**:
```
| COMPETITOR_1_NAME            | @drinkpoppi                        |
| COMPETITOR_1_QR_FILE         | drinkpoppi_top.png                 |
| COMPETITOR_1_QR_URL          | https://www.tiktok.com/@user/...   |
| COMPETITOR_1_QR_VIEWS        | 820K                               |
| COMPETITOR_1_QR_ENGAGEMENT   | 1.5                                |
|                              |                                    |
| COMPETITOR_2_NAME            | @vitalproteins                     |
| COMPETITOR_2_QR_FILE         | vitalproteins_top.png              |
| COMPETITOR_2_QR_URL          | https://www.tiktok.com/@user/...   |
| COMPETITOR_2_QR_VIEWS        | 720K                               |
| COMPETITOR_2_QR_ENGAGEMENT   | 1.4                                |
|                              |                                    |
| COMPETITOR_3_NAME            | @nike                              |
| COMPETITOR_3_QR_FILE         | nike_top.png                       |
| COMPETITOR_3_QR_URL          | https://www.tiktok.com/@user/...   |
| COMPETITOR_3_QR_VIEWS        | 650K                               |
| COMPETITOR_3_QR_ENGAGEMENT   | 1.3                                |
```

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
