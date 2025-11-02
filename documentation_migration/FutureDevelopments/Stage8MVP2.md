# Stage 8 MVP: LLM-Optimized Implementation Guide  

**Purpose**: Consolidated, self-contained implementation specifications for LLM-driven development

**Parent Document**: Stage8MVP.md (architectural overview and shared context)

**Usage**: Give each Section 3.X to an LLM in isolation to implement that specific extraction script

---

## Section 0: Architecture & Prerequisites

### 0.1 Dependency on ML Pipeline

**CRITICAL**: These extraction scripts require completed ML pipeline runs (Stages 1-7).

**Input Dependencies**:
- **Stage 1 (Video Discovery)**: `winner_analysis.json`, `selection_manifest.json`, `selected_videos.json`
- **Stage 2.7 (Content Analysis)**: Content classification JSONs (analysis-level, shared across buckets)
- **Stage 7 (LLM Analysis)**: `winning_formulas.json` per bucket (Report 2 only)

**Prerequisites Per Report**:
- **Report 1 (Hashtag → Client)**: Stages 1, 2.7, 7 complete
- **Report 2 (Hashtag → Creator)**: Stages 1, 2.7, 7 complete
- **Report 3 (Single Competitor)**: Stages 1, 2.7 complete
- **Report 4 (Multi-Competitor)**: Stages 1, 2.7 complete for all competitors

---

### 0.2 Actual Directory Structure (Real Output)

**Base Path Template**:
```
/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/
```

**Complete Structure** (based on real ML pipeline output):
```
/data/clients/{client_id}/hashtags/{target}/{mode}_{strategy}/
├── config.json                          # Run configuration
├── winner_analysis.json                 # Top 3 winning buckets identification
├── selection_manifest.json              # Video IDs per bucket (top/bottom performers)
├── content_analysis/
│   ├── validated/                       # 🆕 Per-bucket validated classifications
│   │   ├── bucket_18-33s/
│   │   │   ├── {video_id}_content.json # Includes "performer_type" & "bucket" fields
│   │   │   └── ... (82 files)
│   │   ├── bucket_33-60s/
│   │   │   └── ... (70 files)
│   │   └── bucket_60-90s/
│   │       └── ... (89 files)
│   └── raw_llm_output/                  # Unprocessed LLM responses
│       ├── bucket_18-33s/
│       ├── bucket_33-60s/
│       └── bucket_60-90s/
├── content_taxonomies/
│   ├── {hashtag}_raw_discovery.json    # Analysis-level aggregation (48 sample videos)
│   └── {hashtag}_taxonomy.json         # Category definitions
├── checkpoints/                         # Processing state
└── buckets/
    ├── bucket_18-33s/
    │   ├── selected_videos.json         # TikTok API metadata for this bucket
    │   ├── videos/                      # Raw MP4s
    │   ├── analysis/
    │   │   ├── insights/                # Temporal windows JSON (Stage 2)
    │   │   ├── unified/                 # Intermediate timeline + ML data
    │   │   └── service_debug/           # ML service outputs
    │   ├── validation/                  # Pipeline validation
    │   ├── ml_analysis/
    │   │   ├── aggregated_features.csv  # Stage 3 output
    │   │   ├── rf_transformed.csv       # Stage 4 output
    │   │   ├── km_transformed.csv       # Stage 4 output
    │   │   ├── {window}_rf_analysis.json      # Stage 6 output
    │   │   ├── {window}_kmeans_analysis.json  # Stage 6 output
    │   │   └── llm/                     # ⚠️ Stage 7 LLM outputs
    │   │       ├── hook_analysis.json
    │   │       ├── middle_1_analysis.json
    │   │       ├── middle_2_analysis.json
    │   │       ├── closing_analysis.json
    │   │       ├── complete_analysis_18-33s.json
    │   │       └── winning_formulas.json  # ⚠️ Used by Report 2
    │   ├── models/                      # Trained models (.pkl)
    │   ├── checkpoints/
    │   ├── logs/
    │   ├── reports/                     # Empty (PDFs not generated yet)
    │   └── flagged_videos/
    ├── bucket_13-18s/
    └── bucket_60-90s/
```

**Key Path Differences from Idealized Documentation**:
1. **content_analysis/validated/** is organized **per-bucket** (new refactored structure)
2. Each validated file includes `"performer_type": "top"|"bottom"` field for filtering
3. Each validated file includes `"bucket": "18-33s"` field for self-documentation
4. **LLM outputs** are in `ml_analysis/llm/`, not `llm_reports/analysis/`
5. **LLM file naming**: `{window}_analysis.json`, `winning_formulas.json` (not `call_1_*`, `insights.json`)

---

### 0.3 Content Analysis Data Architecture

**Stage 2.6 Discovery (Sample-Based)**:
- **Processes**: 48 sample videos across all buckets
- **Output**: `content_taxonomies/{hashtag}_raw_discovery.json` (analysis-level aggregation)
- **Purpose**: Pattern discovery, taxonomy creation for Stage 2.6 ONLY
- **Used by Stage 8?**: ❌ NO - Stage 8 reports use full 300 videos from Stage 2.7

**Stage 2.7 Classification (Full Dataset)**:
- **Processes**: 300+ total videos across all buckets
- **Output**: `content_analysis/validated/bucket_{name}/*_content.json` (per-bucket organization)
- **Purpose**: Complete video classification for ALL Stage 8 reports
- **Used by**: ALL Stage 8 reports (with different aggregation strategies)

**Key Distinction**:
- **`raw_discovery.json`**: 48 sample videos - ONLY for Stage 2.6 discovery, NOT for Stage 8 reports
- **`validated/` files**: 300+ full videos - PRIMARY data source for ALL Stage 8 reports

**Stage 8 Aggregation Strategy Matrix**:

| Report | Aggregation Strategy | Data Source | Method |
|--------|---------------------|-------------|--------|
| Report 1 (Client) | All-buckets | `validated/` across ALL buckets | `aggregate_content_classifications()` + combine Counters |
| Report 2 (Creator) | Per-bucket | `validated/bucket_{name}/` | `aggregate_content_classifications()` per bucket |
| Report 3 (Competitor) | All-buckets | `validated/` across ALL buckets | `aggregate_content_classifications()` + combine Counters |
| Report 4 (Multi-Competitor) | Per-bucket per-competitor | `validated/bucket_{name}/` | `aggregate_content_classifications()` per bucket per competitor |

**All-Buckets vs Per-Bucket**:
- **All-buckets**: Aggregate ALL 300 videos together (combine Counters from multiple buckets)
- **Per-bucket**: Keep bucket aggregations separate (different patterns per duration)

**Reading validated Files**:

Stage 2.7 classification outputs are organized **per-bucket** in `validated/` subdirectories. Each file contains:
- `"performer_type": "top"` or `"performer_type": "bottom"` - for filtering
- `"bucket": "18-33s"` - bucket identifier (self-documenting)
- All 12 classification fields (content_category, hook_strategy, caption_analysis, etc.)

**Example Direct Path Access**:
```python
# Direct path to bucket's validated files
base_path = "/data/clients/acme/hashtags/nutrition/top_contrastive"
bucket_name = "18-33s"

bucket_dir = f"{base_path}/content_analysis/validated/bucket_{bucket_name}"
content_files = glob.glob(f"{bucket_dir}/*_content.json")

# Filter by performer_type field in each JSON
for file_path in content_files:
    with open(file_path) as f:
        content = json.load(f)

    if content.get("performer_type") == "top":
        # Process this top performer video
        ...
```

---

### 0.4 Output Paths (Stage 8 MVP Scripts)

**Report 1 (Hashtag → Client)**:
```
/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/
└── {target}_client_data.xlsx
```

**Report 2 (Hashtag → Creator)**:
```
/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/
├── {target}_creator_data.xlsx (3 tabs)
└── qr_codes/
    ├── {target}_{bucket1}_top.png
    ├── {target}_{bucket1}_bottom.png
    ├── {target}_{bucket2}_top.png
    ├── {target}_{bucket2}_bottom.png
    ├── {target}_{bucket3}_top.png
    └── {target}_{bucket3}_bottom.png
```

**Report 3 (Single Competitor)**:
```
/data/clients/{client}/competitors/{competitor}/{mode}_{strategy}/
├── {competitor}_analysis_data.xlsx
└── qr_codes/
    └── {competitor}_top.png
```

**Report 4 (Multi-Competitor)**:
```
/data/clients/{client}/market_intelligence/multi_competitor/
├── market_intelligence_report.xlsx
└── qr_codes/
    ├── {competitor1}_top.png
    ├── {competitor2}_top.png
    └── {competitor3}_top.png
```

---

## Section 3.1: `extract_creator_data.py` - COMPLETE IMPLEMENTATION GUIDE

### Overview

**Purpose**: Extract 3 creative formulas (one per winning bucket) from Stage 7 analysis and generate Report 2 (Hashtag → Creator)

**Report Type**: Report 2 from Stage8MVP_Reports.md Section 2

**Deliverable**: 1 Excel file with 3 tabs + 6 QR code images

**CLI Usage**:
```bash
python extract_creator_data.py --client acme --hashtag nutrition --mode top --strategy contrastive
```

**Output Files**:
```
/data/clients/acme/hashtags/nutrition/top_contrastive/
├── nutrition_creator_data.xlsx (3 tabs: one per winning bucket formula)
└── qr_codes/
    ├── nutrition_18-33s_top.png
    ├── nutrition_18-33s_bottom.png
    ├── nutrition_13-18s_top.png
    ├── nutrition_13-18s_bottom.png
    ├── nutrition_60-90s_top.png
    └── nutrition_60-90s_bottom.png
```

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

---

### Complete Field List (Per Tab)

**Excel Structure**: 3 tabs (one per winning bucket formula)
- Tab 1: `Formula_18-33s`
- Tab 2: `Formula_13-18s`
- Tab 3: `Formula_60-90s`

**Each tab contains the same ~35 fields** (values are bucket-specific):

```python
# Field structure per tab - two-column format: Field Name | Value
fields_per_tab = [
    # =============================
    # PAGE 1: WHY THIS WORKS
    # =============================
    ('PAGE_1_WHY_THIS_WORKS', ''),  # Section divider
    ('', ''),  # Empty row

    # --- Header Section ---
    ('DURATION', '18-33s'),  # Bucket name
    ('HASHTAG', '#nutrition'),  # From CLI parameter

    # --- The Proof: Performance Comparison ---
    ('', ''),
    ('TOP_CLUSTER_AVG_VIEWS', '620K'),  # From calculate_proof_metrics_bucket_scoped()
    ('TOP_CLUSTER_AVG_ENG', '1.2'),
    ('TOP_CLUSTER_INTERACTIONS', '7440'),
    ('', ''),
    ('BOTTOM_CLUSTER_AVG_VIEWS', '380K'),
    ('BOTTOM_CLUSTER_AVG_ENG', '0.8'),
    ('BOTTOM_CLUSTER_INTERACTIONS', '3040'),
    ('', ''),
    ('VIEW_MULTIPLIER', '1.6x'),  # Calculated: top_views / bottom_views
    ('VIEW_INCREASE_PCT', '63'),  # Calculated: ((top - bottom) / bottom) * 100
    ('ENG_MULTIPLIER', '1.5x'),  # Calculated: top_eng / bottom_eng
    ('ENG_INCREASE_PCT', '50'),  # Calculated: ((top - bottom) / bottom) * 100

    # =============================
    # PAGE 2: HOW TO EXECUTE
    # =============================
    ('', ''),
    ('PAGE_2_HOW_TO_EXECUTE', ''),
    ('', ''),

    # --- Freestyle Tips ---
    ('VIDEO_CATEGORY', 'Recipe Tutorial'),  # From Stage 2.7 content_analysis (most common)

    # --- Phase 1: Hook (0-3s) ---
    ('', ''),
    ('PHASE_1_LABEL', '--- Phase 1: Hook (0-3s) ---'),
    ('PHASE_1_TIMING', '0-3s'),
    ('PHASE_1_CONTENT_PATTERN', 'Problem-Solution'),  # From aggregate_content_classifications()

    # --- Phase 2: Middle (3s to last 3s) ---
    ('', ''),
    ('PHASE_2_LABEL', '--- Phase 2: Middle (3s to last 3s) ---'),
    ('PHASE_2_TIMING', '3s to last 3s'),
    ('PHASE_2_KEYWORD_1', 'gut health'),  # From aggregate_content_classifications() -> keywords
    ('PHASE_2_KEYWORD_2', 'protein'),
    ('PHASE_2_KEYWORD_3', 'anti-inflammatory'),
    ('PHASE_2_TACTIC_1', 'Personal testimony'),  # From aggregate_content_classifications() -> content_tactics
    ('PHASE_2_TACTIC_2', 'Before/after reveal'),
    ('', ''),
    ('SUPPLEMENTARY_INSIGHT_1', 'middle_3_eye_contact_rate: 0.57 in top vs 0.43 in bottom (gap: 0.14)'),  # From winning_formulas.json -> supplementary_insights.universal_principles
    ('SUPPLEMENTARY_INSIGHT_2', 'middle_1_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)'),
    ('SUPPLEMENTARY_INSIGHT_3', 'middle_3_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)'),
    ('SUPPLEMENTARY_INSIGHT_4', 'middle_3_energy_level: 0.10 in top vs 0.06 in bottom (gap: 0.04)'),
    ('SUPPLEMENTARY_INSIGHT_5', 'hook_eye_contact_rate: 0.51 in top vs 0.63 in bottom (gap: 0.11)'),

    # --- Phase 3: Closing (last 3s) ---
    ('', ''),
    ('PHASE_3_LABEL', '--- Phase 3: Closing (last 3s) ---'),
    ('PHASE_3_TIMING', 'last 3s'),
    ('PHASE_3_CTA_TYPE_1', 'link_in_bio'),  # From get_top_n_from_field(field="caption_cta_type", n=3)
    ('PHASE_3_CTA_DESC_1', 'Direct viewers to link in bio'),  # From get_descriptions_from_taxonomy()
    ('PHASE_3_CTA_TYPE_2', 'save_post'),
    ('PHASE_3_CTA_DESC_2', 'Encourage saving post for later'),
    ('PHASE_3_CTA_TYPE_3', 'comment'),
    ('PHASE_3_CTA_DESC_3', 'Ask viewers to comment'),

    # --- Caption Structure ---
    ('', ''),
    ('CAPTION_HOOK_TYPE', 'question'),  # From aggregate_content_classifications() -> caption_analysis.hook_type
    ('CAPTION_LENGTH', 'short'),  # From aggregate_content_classifications() -> caption_analysis.caption_length
    ('CAPTION_EMOJI_USAGE', 'some'),  # From aggregate_content_classifications() -> caption_analysis.emoji_usage
    ('CAPTION_HASHTAG_COUNT', '7'),  # From aggregate_content_classifications() -> caption_analysis.hashtag_count (mean)

    # --- Ready Templates ---
    ('', ''),
    ('TEMPLATE_1_NAME', 'The Silent-to-Vocal Engagement Journey'),  # From winning_formulas.json -> creative_reports[0].formula_name
    ('TEMPLATE_1_HOOK', 'Hook (0-3s): Strong eye contact (0.77), prominent face presence (0.42), establish direct connection'),  # From step_by_step_template
    ('TEMPLATE_1_MIDDLE', 'Middle_1 (3-6s): Transition to pure visual storytelling (0.00 speech), let visuals speak'),  # First Middle line
    ('TEMPLATE_1_CLOSING', 'Closing (23-26s): Visual-first silent closer, minimal verbal content (0.09), indirect gaze (0.19)'),  # From step_by_step_template
    ('', ''),
    ('TEMPLATE_2_NAME', 'The Visual Storytelling Formula'),  # From creative_reports[1].formula_name
    ('TEMPLATE_2_HOOK', 'Hook: Use multiple visual angles or dynamic elements to create immediate visual interest'),
    ('TEMPLATE_2_MIDDLE', 'Middle: Maintain visual variety with strategic scene transitions and visual enhancements'),
    ('TEMPLATE_2_CLOSING', 'Closing: Return to primary visual focus while maintaining dynamic elements'),
    ('', ''),
    ('TEMPLATE_3_NAME', 'The Vocal Variety Formula'),  # From creative_reports[2].formula_name
    ('TEMPLATE_3_HOOK', 'Hook: Establish vocal tone with clear articulation and moderate pacing'),
    ('TEMPLATE_3_MIDDLE', 'Middle: Use strategic pauses and vocal variety for emphasis and engagement'),
    ('TEMPLATE_3_CLOSING', 'Closing: Maintain vocal energy while delivering clear call-to-action'),

    # --- QR Codes ---
    ('', ''),
    ('QR_CODE_TOP_FILE', 'nutrition_18-33s_top.png'),  # From generate_qr_codes()
    ('QR_CODE_TOP_URL', 'https://www.tiktok.com/@agitthaiii/video/7545713916584774968'),
    ('QR_CODE_TOP_VIEWS', '620K'),
    ('', ''),
    ('QR_CODE_BOTTOM_FILE', 'nutrition_18-33s_bottom.png'),
    ('QR_CODE_BOTTOM_URL', 'https://www.tiktok.com/@ahealthydoseofash/video/7560886598309612814'),
    ('QR_CODE_BOTTOM_VIEWS', '95K'),
]
```

**Notes**:
- Total fields per tab: ~85 (including section dividers and empty rows)
- Field naming: `UPPERCASE_WITH_UNDERSCORES`
- Multi-value fields use numbered suffixes (e.g., `KEYWORD_1`, `KEYWORD_2`, `KEYWORD_3`)
- Empty rows (`('', '')`) provide visual separation between sections
- Ready Templates: 12 fields (3 templates × 4 fields each: NAME, HOOK, MIDDLE, CLOSING)
- Supplementary Insights: 5 fields (quantitative metrics from Stage 7 LLM analysis)
- Section dividers use equals signs for page-level, dashes for subsections

---

### Required Functions

This section defines all functions needed for `extract_creator_data.py`. Functions are documented inline for self-contained LLM implementation.

---

#### Function 1: `calculate_proof_metrics_bucket_scoped()`

**Purpose**: Calculate performance comparison metrics for "The Proof" section with bucket-scoping

**Used by**: Report 2 only

**Why bucket-scoping matters**: Report 2 has 3 formulas (one per winning bucket). "The Proof" section compares videos using the pattern vs not using it WITHIN the same duration bucket. Without bucket-scoping, metrics would mix all durations (9s, 18s, 33s, 60s), making comparison invalid.

**Input Parameters**:
- `bucket_path` (str): Path to bucket folder
  Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/`
- `bucket_name` (str): Duration bucket name
  Example: `"18-33s"`, `"60-90s"`
- `formula_cluster_id` (int): Winning cluster ID from Stage 7
  Example: 0, 1, or 2 (from K-means clustering)

**Output**:
```python
{
    'top_cluster': {
        'avg_views': 620000,
        'avg_engagement': 1.2,
        'avg_interactions': 7440,
        'video_count': 15
    },
    'bottom_cluster': {
        'avg_views': 380000,
        'avg_engagement': 0.8,
        'avg_interactions': 3040,
        'video_count': 25
    }
}
```

**Implementation**:
```python
import json
import os

def calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name, formula_cluster_id):
    """
    Calculate performance comparison for top cluster vs bottom cluster within ONE bucket.

    Process:
    1. Load K-means cluster assignments (Stage 6)
    2. Load selection_manifest.json to get top_performers for THIS BUCKET
    3. Filter videos by: cluster membership AND bucket's top_performers
    4. Calculate avg views and engagement for top cluster
    5. Calculate avg views and engagement for bottom cluster
    6. Return comparison metrics
    """

    # Step 1: Load K-means cluster assignments
    kmeans_path = os.path.join(bucket_path, 'ml_analysis', 'hook_kmeans_analysis.json')
    with open(kmeans_path, 'r') as f:
        kmeans_data = json.load(f)

    # Get video IDs in winning cluster
    winning_cluster_video_ids = set()
    other_cluster_video_ids = set()

    for cluster in kmeans_data['clusters']:
        for video in cluster['videos']:
            if cluster['cluster_id'] == formula_cluster_id:
                winning_cluster_video_ids.add(video['video_id'])
            else:
                other_cluster_video_ids.add(video['video_id'])

    # Step 2: Load selection manifest to filter by top_performers
    manifest_path = os.path.join(bucket_path, '..', '..', 'selection_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Get top_performers for this bucket
    bucket_top_performers = set(manifest['videos_by_bucket'][bucket_name]['top_performers'])

    # Step 3: Load selected_videos.json for video metadata
    selected_videos_path = os.path.join(bucket_path, 'selected_videos.json')
    with open(selected_videos_path, 'r') as f:
        selected_videos = json.load(f)

    # Step 4: Calculate metrics for top cluster (winning cluster + top performers)
    top_cluster_videos = []
    for video in selected_videos['videos']:
        if video['video_id'] in winning_cluster_video_ids and video['video_id'] in bucket_top_performers:
            top_cluster_videos.append(video)

    # Step 5: Calculate metrics for bottom cluster (other clusters + top performers)
    bottom_cluster_videos = []
    for video in selected_videos['videos']:
        if video['video_id'] in other_cluster_video_ids and video['video_id'] in bucket_top_performers:
            bottom_cluster_videos.append(video)

    # Step 6: Calculate averages
    def calc_avg_metrics(videos):
        if not videos:
            return {'avg_views': 0, 'avg_engagement': 0, 'avg_interactions': 0, 'video_count': 0}

        total_views = sum(v['playCount'] for v in videos)

        # Calculate engagement using calculate_engagement_metrics() - See Function 6
        total_engagement = 0
        for v in videos:
            engagement = calculate_engagement_metrics(v)
            total_engagement += engagement

        avg_interactions = sum(
            v.get('diggCount', 0) + v.get('commentCount', 0) +
            v.get('shareCount', 0) + v.get('collectCount', 0)
            for v in videos
        ) / len(videos)

        return {
            'avg_views': total_views / len(videos),
            'avg_engagement': total_engagement / len(videos),
            'avg_interactions': avg_interactions,
            'video_count': len(videos)
        }

    return {
        'top_cluster': calc_avg_metrics(top_cluster_videos),
        'bottom_cluster': calc_avg_metrics(bottom_cluster_videos)
    }
```

**Data Sources**:
- K-means analysis: `{bucket_path}/ml_analysis/hook_kmeans_analysis.json`
- Selection manifest: `{base_path}/selection_manifest.json`
- Video metadata: `{bucket_path}/selected_videos.json`

---

#### Function 2: `select_qr_code_videos()`

**Purpose**: Select top and bottom performer videos for QR code generation

**Used by**: Reports 2, 3, 4

**Input Parameters**:
- `bucket_path` (str): Path to bucket folder
- `bucket_name` (str): Duration bucket name

**Output**:
```python
{
    'top_performer': {
        'video_id': '7545713916584774968',
        'url': 'https://www.tiktok.com/@agitthaiii/video/7545713916584774968',
        'views': 620000,
        'createTime': 1735689600
    },
    'bottom_performer': {
        'video_id': '7560886598309612814',
        'url': 'https://www.tiktok.com/@ahealthydoseofash/video/7560886598309612814',
        'views': 95000,
        'createTime': 1734480000
    }
}
```

**Implementation**:
```python
import json
import os

def select_qr_code_videos(bucket_path, bucket_name):
    """
    Select top and bottom performer videos for QR codes.

    Selection criteria:
    - Top: Highest view count from top_performers
    - Bottom: Highest view count from bottom_performers (for stability - less likely deleted)

    Process:
    1. Load selection_manifest.json to get performer video IDs
    2. Load selected_videos.json to get video metadata
    3. Select max by (playCount, createTime) - newer videos preferred if views tied
    """

    # Step 1: Load selection manifest
    manifest_path = os.path.join(bucket_path, '..', '..', 'selection_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    top_performer_ids = set(manifest['videos_by_bucket'][bucket_name]['top_performers'])
    bottom_performer_ids = set(manifest['videos_by_bucket'][bucket_name]['bottom_performers'])

    # Step 2: Load selected videos
    selected_videos_path = os.path.join(bucket_path, 'selected_videos.json')
    with open(selected_videos_path, 'r') as f:
        selected_videos = json.load(f)

    # Step 3: Filter and select
    top_performers = [v for v in selected_videos['videos'] if v['video_id'] in top_performer_ids]
    bottom_performers = [v for v in selected_videos['videos'] if v['video_id'] in bottom_performer_ids]

    # Select max by playCount (primary), then createTime (secondary - newer = more stable)
    top_video = max(top_performers, key=lambda v: (v['playCount'], v['createTime']))
    bottom_video = max(bottom_performers, key=lambda v: (v['playCount'], v['createTime']))

    return {
        'top_performer': {
            'video_id': top_video['video_id'],
            'url': top_video['webVideoUrl'],
            'views': top_video['playCount'],
            'createTime': top_video['createTime']
        },
        'bottom_performer': {
            'video_id': bottom_video['video_id'],
            'url': bottom_video['webVideoUrl'],
            'views': bottom_video['playCount'],
            'createTime': bottom_video['createTime']
        }
    }
```

**Data Sources**:
- Selection manifest: `{base_path}/selection_manifest.json`
- Video metadata: `{bucket_path}/selected_videos.json`

---

#### Function 3: `generate_qr_codes()`

**Purpose**: Generate QR code PNG files from TikTok video URLs

**Used by**: Reports 2, 3, 4

**Input Parameters**:
- `video_url` (str): TikTok video URL to encode
- `output_path` (str): Full path where to save PNG file
  Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/qr_codes/nutrition_18-33s_top.png`

**Output**: None (writes PNG file to disk)

**Implementation**:
```python
import qrcode
import os

def generate_qr_codes(video_url, output_path):
    """
    Generate QR code PNG from TikTok URL.

    QR Code Specifications:
    - Size: ~290x290 pixels (1" x 1" at standard DPI)
    - Error correction: Medium (15% damage tolerance)
    - Format: PNG, black on white
    - File size: ~5KB per QR code

    Library: qrcode[pil] (free, BSD license)
    Install: pip install qrcode[pil]
    """

    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,  # Auto-size (1-40, where 1 is smallest)
        error_correction=qrcode.constants.ERROR_CORRECT_M,  # Medium (15% tolerance)
        box_size=10,  # Pixel size per box (10 = ~290x290 pixels for small QR)
        border=4  # Border size in boxes (4 is standard minimum)
    )

    # Add data and generate
    qr.add_data(video_url)
    qr.make(fit=True)

    # Create image
    img = qr.make_image(fill_color="black", back_color="white")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save PNG
    img.save(output_path)
```

**Dependencies**:
- `qrcode[pil]` - QR code generation library (includes Pillow for image creation)
- Already installed via: `pip install qrcode[pil]`

**Why direct URLs (no tracking)**:
- Simplest for MVP (no API keys, no external service)
- Zero cost (vs Bitly $10+/month for analytics)
- Zero maintenance (no service downtime risk)
- Can add tracking later if needed by regenerating QR codes

---

#### Function 4: `aggregate_content_classifications()`

**Purpose**: Aggregate Stage 2.7 content analysis classifications across videos in a bucket

**Used by**: Reports 1, 2

**Input Parameters**:
- `bucket_path` (str): Path to bucket folder
  Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/`
- `performer_type` (str): "top" or "bottom" - which performers to analyze

**Output**:
```python
{
    'content_category': {
        'all_values': ['recipe_tutorial', 'wellness_practice', 'recipe_tutorial', ...],  # All 40 values
        'top_3': [
            ('recipe_tutorial', 38),  # (name, percentage)
            ('wellness_practice', 28),
            ('supplement_review', 22)
        ],
        'most_common': 'recipe_tutorial'
    },
    'hook_strategy': {
        'all_values': ['problem_solution', 'question_hook', ...],
        'top_3': [
            ('problem_solution', 42),
            ('question_hook', 35),
            ('direct_statement', 23)
        ],
        'most_common': 'problem_solution'
    },
    'keywords': {
        'all_values': ['#guthealth', '#protein', '#guthealth', ...],  # Flattened from arrays
        'top_3': ['#guthealth', '#protein', '#antiinflammatory'],
        'top_5': ['#guthealth', '#protein', '#antiinflammatory', '#metabolism', '#fiber']
    },
    'pain_points': {
        'all_values': ['bloating', 'low_energy', ...],
        'top_3': [
            ('Bloating', 48),
            ('Low Energy', 42),
            ('Inflammation', 38)
        ]
    },
    'content_tactics': {
        'all_values': ['personal_testimony', 'before_after_reveal', ...],
        'top_2': ['personal_testimony', 'before_after_reveal']
    },
    'caption_analysis': {
        'hook_type': {
            'most_common': 'question',
            'percentage': 58
        },
        'cta_type': {
            'most_common': 'link_in_bio',
            'percentage': 58
        },
        'caption_length': {
            'most_common': 'short',
            'percentage': 68
        },
        'emoji_usage': {
            'most_common': 'some',
            'percentage': 72
        },
        'hashtag_count': {
            'mean': 7.2,
            'rounded': 7
        }
    }
}
```

**Implementation**:
```python
import json
import os
from collections import Counter

def aggregate_content_classifications(bucket_name, base_path, performer_type="top"):
    """
    Aggregate content patterns from per-bucket validated files.

    NEW APPROACH (Post-Refactor):
    - Files organized by bucket in validated/ subdirectory
    - Each file has "performer_type" field for filtering
    - Direct path to bucket directory (no navigation needed)
    - No need to load selected_videos.json

    Process:
    1. Build direct path to bucket's validated directory
    2. Load all content files in bucket
    3. Filter by performer_type field in JSON
    4. Aggregate all classification fields
    5. Return Counter objects for downstream processing

    Args:
        bucket_name: Bucket identifier (e.g., "18-33s")
        base_path: Analysis base path (e.g., "/data/clients/acme/hashtags/nutrition/top_contrastive")
        performer_type: "top" or "bottom" (default: "top")

    Returns:
        dict: {
            "content_category": Counter({"wellness_routine": 25, ...}),
            "hook_strategy": Counter({"question": 18, ...}),
            "pain_points": Counter({"bloating": 12, ...}),
            "keywords": Counter({"gut health": 15, ...}),
            "engagement_drivers": Counter({"before_after": 8, ...}),
            "content_tactics": Counter({"direct_camera": 22, ...}),
            "caption_hook_type": Counter({"statement": 45, ...}),
            "caption_cta_type": Counter({"link_in_bio": 18, ...}),
            "hashtag_counts": [0, 3, 5, 7, ...],
            "processed_count": 82
        }
    """
    import os
    import json
    import glob
    from collections import Counter

    # Step 1: Build direct path to bucket directory
    bucket_dir = f"{base_path}/content_analysis/validated/bucket_{bucket_name}"

    if not os.path.exists(bucket_dir):
        raise FileNotFoundError(f"Bucket directory not found: {bucket_dir}")

    # Step 2: Load all content files in bucket
    content_files = glob.glob(f"{bucket_dir}/*_content.json")

    if not content_files:
        return {}  # No content files in bucket

    # Step 3: Initialize counters
    content_categories = Counter()
    hook_strategies = Counter()
    closing_strategies = Counter()
    pain_points = Counter()
    keywords = Counter()
    engagement_drivers = Counter()
    content_tactics = Counter()

    # Caption analysis counters
    caption_hook_types = Counter()
    caption_cta_types = Counter()
    hashtag_counts = []

    # Step 4: Aggregate from files matching performer_type
    processed_count = 0

    for file_path in content_files:
        with open(file_path) as f:
            content = json.load(f)

        # Filter by performer_type field (built into each JSON)
        if content.get("performer_type") != performer_type:
            continue  # Skip non-matching performers

        processed_count += 1

        # Single-value fields
        if content.get("content_category"):
            content_categories[content["content_category"]] += 1
        if content.get("hook_strategy"):
            hook_strategies[content["hook_strategy"]] += 1
        if content.get("closing_strategy"):
            closing_strategies[content["closing_strategy"]] += 1

        # Multi-value fields (arrays)
        pain_points.update(content.get("pain_points", []))
        keywords.update(content.get("keywords", []))
        engagement_drivers.update(content.get("engagement_drivers", []))
        content_tactics.update(content.get("content_tactics", []))

        # Caption analysis
        caption = content.get("caption_analysis", {})
        if caption.get("hook_type"):
            caption_hook_types[caption["hook_type"]] += 1
        if caption.get("cta_type"):
            caption_cta_types[caption["cta_type"]] += 1
        if "hashtag_count" in caption:
            hashtag_counts.append(caption["hashtag_count"])

    if processed_count == 0:
        logger.warning(f"No {performer_type} performers found in bucket {bucket_name}")
        return {}

    # Step 5: Return aggregated data as Counters
    return {
        "content_category": content_categories,
        "hook_strategy": hook_strategies,
        "closing_strategy": closing_strategies,
        "pain_points": pain_points,
        "keywords": keywords,
        "engagement_drivers": engagement_drivers,
        "content_tactics": content_tactics,
        "caption_hook_type": caption_hook_types,
        "caption_cta_type": caption_cta_types,
        "hashtag_counts": hashtag_counts,
        "processed_count": processed_count
    }
```

**Data Source**: `/content_analysis/validated/bucket_{name}/*_content.json` (per-bucket organization)

**Filter Field**: `content["performer_type"]` ("top" or "bottom")

**Performance**: ~1.2ms for 82 files (excellent, linear scaling)

**Usage in Report 2**:
```python
# Get top performer content patterns for this bucket
aggregated = aggregate_content_classifications(
    bucket_name="18-33s",
    base_path=base_path,
    performer_type="top"
)

# Extract fields for Excel
phase_1_pattern = aggregated['hook_strategy'].most_common(1)[0][0]  # "problem_solution"
keywords = [k for k, _ in aggregated['keywords'].most_common(3)]  # ["gut health", "protein", "fiber"]
cta_type = aggregated['caption_hook_type'].most_common(1)[0][0]  # "link_in_bio"
video_category = aggregated['content_category'].most_common(1)[0][0]  # "recipe_tutorial"
```

---

#### Function 5: `calculate_engagement_metrics()`

**Purpose**: Calculate real engagement rate from TikTok video metadata

**Used by**: Reports 1, 2, 3, 4 (all reports)

**Note**: This function is used across all reports. Full specification provided here for completeness.

**Input**: Video metadata dictionary with engagement fields

**Output**: Engagement rate as float (percentage)

**Implementation**:
```python
def calculate_engagement_metrics(video_metadata):
    """
    Calculate engagement rate from TikTok video metadata.

    Formula: (likes + comments + shares + saves) / views × 100

    Input fields (from Apify unified_analysis JSON):
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

**Data Source**: `{video_id}_unified_analysis.json` → `metadata` object (lines 8-12 in Apify output)

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

### Inline Calculations

These are simple calculations that don't need separate functions:

#### Format Views with K/M Suffix

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

#### Calculate Multipliers

```python
def calculate_multiplier(top_value, bottom_value):
    """
    Calculate multiplier for "X.Xx more" format.

    Example: 620K / 380K = 1.6x
    """
    if bottom_value == 0:
        return "N/A"
    multiplier = top_value / bottom_value
    return f"{multiplier:.1f}x"
```

#### Calculate Percentage Increase

```python
def calculate_percentage_increase(top_value, bottom_value):
    """
    Calculate percentage increase for "X% higher" format.

    Example: ((620K - 380K) / 380K) * 100 = 63%
    """
    if bottom_value == 0:
        return "N/A"
    increase = ((top_value - bottom_value) / bottom_value) * 100
    return str(int(increase))
```

---

### Data Source File Formats

This section documents the exact structure of all JSON files referenced by the functions above. Use these structures to correctly access fields when implementing the script.

---

#### File 1: `winner_analysis.json`

**Location**: `{base_path}/winner_analysis.json`
- Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json`

**Purpose**: Identifies the 3 winning duration buckets for this hashtag analysis

**Structure**:
```python
{
    "top_3_buckets": ["18-33s", "13-18s", "60-90s"],  # Always 3 buckets
    "bucket_distribution": {
        "0-3s": 145,
        "3-9s": 218,
        "9-13s": 273,
        "13-18s": 401,  # Winning bucket #2
        "18-33s": 511,  # Winning bucket #1
        "33-60s": 219,
        "60-90s": 201,  # Winning bucket #3
        "90-120s": 18
    },
    "top_100_distribution": {
        "18-33s": 43,  # 43% of top performers are in this bucket
        "13-18s": 12,
        "60-90s": 11
        # ... other buckets
    }
}
```

**Used for fields**: `DURATION` (one per tab - extracted from `top_3_buckets`)

---

#### File 2: `winning_formulas.json`

**Location**: `{bucket_path}/ml_analysis/llm/winning_formulas.json`
- Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/ml_analysis/llm/winning_formulas.json`

**Purpose**: Contains Stage 7 LLM analysis of the winning creative formula for this bucket

**Structure**:
```python
{
    "bucket": "18-33s",
    "cluster_id": 0
}
```

---

#### File 3: `selection_manifest.json`

**Location**: `{base_path}/selection_manifest.json`
- Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/selection_manifest.json`

**Purpose**: Lists which video IDs are top performers vs bottom performers per bucket

**Structure**:
```python
{
    "videos_by_bucket": {
        "18-33s": {
            "top_performers": [
                "7545713916584774968",
                "7560886598309612814",
                # ... ~40 video IDs
            ],
            "bottom_performers": [
                "7523456789012345678",
                "7534567890123456789",
                # ... ~20 video IDs
            ]
        },
        "13-18s": {
            "top_performers": [...],
            "bottom_performers": [...]
        },
        "60-90s": {
            "top_performers": [...],
            "bottom_performers": [...]
        }
    }
}
```

**Used by**:
- `select_qr_code_videos()` - to get video IDs for QR code selection
- `aggregate_content_classifications()` - to filter which videos to analyze

---

#### File 4: `selected_videos.json`

**Location**: `{bucket_path}/selected_videos.json`
- Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/selected_videos.json`

**Purpose**: Contains metadata for all videos in this bucket (both top and bottom performers)

**Structure**:
```python
{
    "videos": [
        {
            "video_id": "7545713916584774968",
            "webVideoUrl": "https://www.tiktok.com/@agitthaiii/video/7545713916584774968",
            "playCount": 620000,
            "diggCount": 5580,  # likes
            "commentCount": 1240,
            "shareCount": 310,
            "collectCount": 310,  # saves/bookmarks
            "createTime": 1735689600,
            "is_top_performer": true
        },
        {
            "video_id": "7560886598309612814",
            "webVideoUrl": "https://www.tiktok.com/@ahealthydoseofash/video/7560886598309612814",
            "playCount": 95000,
            "diggCount": 760,
            "commentCount": 95,
            "shareCount": 38,
            "collectCount": 57,
            "createTime": 1734480000,
            "is_top_performer": false
        }
        // ... more videos
    ]
}
```

**Used for fields**:
- `QR_CODE_TOP_URL`, `QR_CODE_TOP_VIEWS` → top performer's `webVideoUrl`, `playCount`
- `QR_CODE_BOTTOM_URL`, `QR_CODE_BOTTOM_VIEWS` → bottom performer's `webVideoUrl`, `playCount`
- Engagement calculation via `calculate_engagement_metrics()`

---

#### File 5: `content_analysis/validated/bucket_{name}/{video_id}_content.json`

**Location**: `{base_path}/content_analysis/validated/bucket_{bucket_name}/{video_id}_content.json` (per-bucket organization)
- Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/content_analysis/validated/bucket_18-33s/7545713916584774968_content.json`

**Purpose**: Stage 2.7 LLM content classification for one video (organized by bucket, includes performer_type field)

**How many files**: Varies per bucket (e.g., bucket_18-33s: 82 files, bucket_33-60s: 70 files)

**Structure**:
```python
{
    "video_id": "7545713916584774968",
    "content_category": "recipe_tutorial",  # Single value (required)
    "hook_strategy": "problem_solution",  # Single value (required)
    "keywords": ["#guthealth", "#protein", "#antiinflammatory"],  # Array (0-N items)
    "pain_points": ["Bloating", "Low Energy"],  # Array (0-N items)
    "engagement_drivers": ["before_after_reveal", "personal_testimony"],  # Array (0-N items)
    "content_tactics": ["personal_story", "direct_to_camera"],  # Array (0-N items)
    "caption_analysis": {
        "hook_type": "question",  # Values: "statement", "question", "command", "teaser"
        "cta_type": "link_in_bio",  # Values: "link_in_bio", "save_post", "comment", "follow", "share", "tag_friend", "none"
        "caption_length": "short",  # Values: "short" (<100 chars), "long" (100+ chars)
        "emoji_usage": "some",  # Values: "none" (0), "some" (1-4), "many" (5+)
        "hashtag_count": 8  # Integer
    },
    "taxonomy_version": "stage2.6_output",
    "confidence": "high",
    "transcript_available": true,
    "bucket": "18-33s",  # 🆕 Bucket identifier
    "performer_type": "top"  # 🆕 "top" or "bottom" for filtering
}
```

**Used for fields** (via `aggregate_content_classifications()`):
- `VIDEO_CATEGORY` → most common `content_category` across top performers
- `PHASE_1_CONTENT_PATTERN` → most common `hook_strategy`
- `PHASE_2_KEYWORD_1-3` → top 3 `keywords` (flattened from all videos)
- `PHASE_2_TACTIC_1-2` → top 2 `content_tactics`
- `PHASE_3_CTA_TYPE` → most common `caption_analysis.cta_type`
- `CAPTION_HOOK_TYPE` → most common `caption_analysis.hook_type`
- `CAPTION_LENGTH` → most common `caption_analysis.caption_length`
- `CAPTION_EMOJI_USAGE` → most common `caption_analysis.emoji_usage`
- `CAPTION_HASHTAG_COUNT` → mean of `caption_analysis.hashtag_count` (rounded)

---

#### File 6: `temporal_windows_updated.json`

**Location**: `/data/clients/{client}/hashtags/{hashtag}/insights/{video_id}_temporal_windows_updated.json`
- Example: `/data/clients/acme/hashtags/nutrition/insights/7545713916584774968_temporal_windows_updated.json`

**Purpose**: RumiAI Stage 2 ML analysis - 60+ quantitative features per temporal segment

**How many files**: One per video (access only for top performers when calculating averages)

**Structure**:
```python
{
    "temporal_windows": {
        "hook": {  # First 3 seconds (0-3s)
            "word_count": 13,
            "energy_level": 0.45,
            "eye_contact_rate": 0.87,  # 0.0-1.0 (proportion of frames with eye contact)
            "average_face_size": 0.44,  # 0.0-1.0 (proportion of frame occupied by face)
            "close_ratio": 0.82,
            "person_count": 1,
            "has_greeting": true,
            "joy_ratio": 0.32
            // ... 50+ more features
        },
        "middle_segments": [  # Varies by video duration (3-5 segments)
            {
                "start_time": 3.0,
                "end_time": 10.0,
                "word_count": 28,
                "energy_level": 0.38,
                "scene_changes": 2,
                "element_count": 25
                // ... more features
            },
            {
                "start_time": 10.0,
                "end_time": 20.0,
                "word_count": 35,
                "energy_level": 0.36,
                "scene_changes": 3,
                "element_count": 30
            }
            // ... more segments depending on duration
        ],
        "closing": {  # Last 3 seconds
            "energy_max": 0.89,
            "has_speech_cta": true,
            "word_count": 8
            // ... more features
        }
    }
}
```

**Used for fields**:
- `PHASE_3_ENERGY_MAX` → Average `temporal_windows.closing.energy_max` across top performers

**Calculation pattern for averages**:
```python
# Example: Calculate average word count in hook across top performers
total_word_count = 0
video_count = 0

for video_id in top_performer_ids:
    temporal_path = f"/data/clients/{client}/hashtags/{hashtag}/insights/{video_id}_temporal_windows_updated.json"
    with open(temporal_path) as f:
        data = json.load(f)

    total_word_count += data['temporal_windows']['hook']['word_count']
    video_count += 1

avg_word_count = total_word_count / video_count
```

---

#### File 7: `hook_kmeans_analysis.json`

**Location**: `{bucket_path}/ml_analysis/hook_kmeans_analysis.json`
- Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/ml_analysis/hook_kmeans_analysis.json`

**Purpose**: Stage 6 K-Means clustering results - shows which videos belong to which cluster

**Structure**:
```python
{
    "clusters": [
        {
            "cluster_id": 0,  # This is the "winning cluster" for this formula
            "videos": [
                {
                    "video_id": "7545713916584774968",
                    "cluster_assignment": 0
                },
                {
                    "video_id": "7560886598309612814",
                    "cluster_assignment": 0
                }
                // ... ~15 videos in this cluster
            ]
        },
        {
            "cluster_id": 1,  # Other cluster
            "videos": [...]
        },
        {
            "cluster_id": 2,  # Other cluster
            "videos": [...]
        }
    ]
}
```

**Used by**: `calculate_proof_metrics_bucket_scoped()` - to separate videos into top cluster (using pattern) vs bottom cluster (not using pattern)

---

### Complete Implementation Pattern

This section shows the full script structure for LLM implementation:

```python
#!/usr/bin/env python3
"""
extract_creator_data.py - Report 2: Hashtag → Creator

Generates 3 creator formula reports (one per winning bucket) with:
- Excel file with 3 tabs
- 6 QR code PNGs (2 per formula)

Usage:
    python extract_creator_data.py --client acme --hashtag nutrition
"""

import argparse
import json
import os
import pandas as pd
import qrcode

# Import inline functions defined above
# (In actual implementation, these would be in the same file or imported from report_utils.py)


def main():
    """Main extraction workflow"""

    # =============================
    # STEP 1: Parse CLI Arguments
    # =============================
    parser = argparse.ArgumentParser(description='Extract Report 2: Hashtag → Creator')
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

    # Load winning buckets
    winner_analysis_path = os.path.join(base_path, 'winner_analysis.json')
    with open(winner_analysis_path) as f:
        winner_data = json.load(f)

    winning_buckets = winner_data['top_3_buckets']  # ['18-33s', '13-18s', '60-90s']
    print(f"Processing {len(winning_buckets)} winning buckets...")

    # =============================
    # STEP 3: Process Each Bucket
    # =============================
    all_tabs = {}  # Store dataframes for each tab
    qr_codes_generated = 0

    for bucket_name in winning_buckets:
        bucket_path = os.path.join(base_path, 'buckets', f'bucket_{bucket_name}')

        # Initialize data list for this tab
        tab_data = []

        # --- PAGE 1: WHY THIS WORKS ---
        tab_data.append(['PAGE_1_WHY_THIS_WORKS', ''])
        tab_data.append(['', ''])

        # Header Section
        tab_data.append(['DURATION', bucket_name])
        tab_data.append(['HASHTAG', f'#{args.hashtag}'])

        # The Proof Section
        tab_data.append(['', ''])
        proof_metrics = calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name, formula_cluster_id=0)

        top_views = proof_metrics['top_cluster']['avg_views']
        top_eng = proof_metrics['top_cluster']['avg_engagement']
        top_interactions = proof_metrics['top_cluster']['avg_interactions']
        bottom_views = proof_metrics['bottom_cluster']['avg_views']
        bottom_eng = proof_metrics['bottom_cluster']['avg_engagement']
        bottom_interactions = proof_metrics['bottom_cluster']['avg_interactions']

        tab_data.append(['TOP_CLUSTER_AVG_VIEWS', format_views(top_views)])
        tab_data.append(['TOP_CLUSTER_AVG_ENG', str(top_eng)])
        tab_data.append(['TOP_CLUSTER_INTERACTIONS', str(int(top_interactions))])
        tab_data.append(['', ''])
        tab_data.append(['BOTTOM_CLUSTER_AVG_VIEWS', format_views(bottom_views)])
        tab_data.append(['BOTTOM_CLUSTER_AVG_ENG', str(bottom_eng)])
        tab_data.append(['BOTTOM_CLUSTER_INTERACTIONS', str(int(bottom_interactions))])
        tab_data.append(['', ''])
        tab_data.append(['VIEW_MULTIPLIER', calculate_multiplier(top_views, bottom_views)])
        tab_data.append(['VIEW_INCREASE_PCT', calculate_percentage_increase(top_views, bottom_views)])
        tab_data.append(['ENG_MULTIPLIER', calculate_multiplier(top_eng, bottom_eng)])
        tab_data.append(['ENG_INCREASE_PCT', calculate_percentage_increase(top_eng, bottom_eng)])

        # --- PAGE 2: HOW TO EXECUTE ---
        tab_data.append(['', ''])
        tab_data.append(['PAGE_2_HOW_TO_EXECUTE', ''])
        tab_data.append(['', ''])

        # Aggregate content patterns for this bucket (top performers only)
        aggregated = aggregate_content_classifications(
            bucket_name=bucket_name,
            base_path=base_path,
            performer_type="top"
        )

        # Video Category
        video_category = aggregated['content_category'].most_common(1)[0][0] if aggregated.get('content_category') else "Unknown"
        tab_data.append(['VIDEO_CATEGORY', video_category])

        # Phase 1: Hook
        tab_data.append(['', ''])
        tab_data.append(['PHASE_1_LABEL', '--- Phase 1: Hook (0-3s) ---'])
        tab_data.append(['PHASE_1_TIMING', '0-3s'])
        phase_1_pattern = aggregated['hook_strategy'].most_common(1)[0][0] if aggregated.get('hook_strategy') else "Unknown"
        tab_data.append(['PHASE_1_CONTENT_PATTERN', phase_1_pattern])

        # Phase 2: Middle
        tab_data.append(['', ''])
        tab_data.append(['PHASE_2_LABEL', '--- Phase 2: Middle (3s to last 3s) ---'])
        tab_data.append(['PHASE_2_TIMING', '3s to last 3s'])
        # Extract top 3 keywords
        top_keywords = [k for k, _ in aggregated['keywords'].most_common(3)] if aggregated.get('keywords') else []
        for i in range(3):
            keyword = top_keywords[i] if i < len(top_keywords) else ""
            tab_data.append([f'PHASE_2_KEYWORD_{i+1}', keyword])
        # Extract top 2 content tactics
        top_tactics = [t for t, _ in aggregated['content_tactics'].most_common(2)] if aggregated.get('content_tactics') else []
        for i in range(2):
            tactic = top_tactics[i] if i < len(top_tactics) else ""
            tab_data.append([f'PHASE_2_TACTIC_{i+1}', tactic])

        # Supplementary Insights
        tab_data.append(['', ''])
        # Load winning_formulas.json to extract supplementary insights
        winning_formulas_path = os.path.join(bucket_path, 'ml_analysis', 'llm', 'winning_formulas.json')

        if os.path.exists(winning_formulas_path):
            with open(winning_formulas_path, 'r') as f:
                winning_formulas = json.load(f)
                universal_principles = winning_formulas.get('supplementary_insights', {}).get('universal_principles', [])

                for i in range(min(5, len(universal_principles))):
                    tab_data.append([f'SUPPLEMENTARY_INSIGHT_{i+1}', universal_principles[i]])

        # Phase 3: Closing
        tab_data.append(['', ''])
        tab_data.append(['PHASE_3_LABEL', '--- Phase 3: Closing (last 3s) ---'])
        tab_data.append(['PHASE_3_TIMING', 'last 3s'])

        # Extract Top 3 CTA types
        top_cta_types = [c for c, _ in aggregated['caption_cta_type'].most_common(3)] if aggregated.get('caption_cta_type') else []
        # TODO: Get CTA descriptions from taxonomy (implement get_descriptions_from_taxonomy if needed)

        for i in range(3):
            cta_type = top_cta_types[i] if i < len(top_cta_types) else ''
            tab_data.append([f'PHASE_3_CTA_TYPE_{i+1}', cta_type])
            tab_data.append([f'PHASE_3_CTA_DESC_{i+1}', ''])  # Placeholder for description

        # Caption Structure
        tab_data.append(['', ''])
        # Extract caption analysis from aggregated data
        caption_hook = aggregated['caption_hook_type'].most_common(1)[0][0] if aggregated.get('caption_hook_type') else "unknown"
        avg_hashtag_count = round(sum(aggregated.get('hashtag_counts', [])) / len(aggregated.get('hashtag_counts', [1]))) if aggregated.get('hashtag_counts') else 0

        tab_data.append(['CAPTION_HOOK_TYPE', caption_hook])
        tab_data.append(['CAPTION_LENGTH', 'short'])  # Placeholder - add to aggregation if needed
        tab_data.append(['CAPTION_EMOJI_USAGE', 'some'])  # Placeholder - add to aggregation if needed
        tab_data.append(['CAPTION_HASHTAG_COUNT', str(avg_hashtag_count)])

        # Ready Templates
        tab_data.append(['', ''])
        # TODO: Load winning_formulas.json from bucket_path
        winning_formulas_path = os.path.join(bucket_path, 'ml_analysis', 'llm', 'winning_formulas.json')

        if os.path.exists(winning_formulas_path):
            with open(winning_formulas_path, 'r') as f:
                winning_formulas = json.load(f)
                creative_reports = winning_formulas.get('creative_reports', [])

                for i in range(min(3, len(creative_reports))):
                    report = creative_reports[i]
                    template_num = i + 1

                    # Extract formula name
                    tab_data.append([f'TEMPLATE_{template_num}_NAME', report.get('formula_name', '')])

                    # Extract step-by-step template
                    steps = report.get('step_by_step_template', [])
                    hook = next((s for s in steps if s.startswith('Hook')), '')
                    middle = next((s for s in steps if s.startswith('Middle')), '')
                    closing = next((s for s in steps if s.startswith('Closing')), '')

                    tab_data.append([f'TEMPLATE_{template_num}_HOOK', hook])
                    tab_data.append([f'TEMPLATE_{template_num}_MIDDLE', middle])
                    tab_data.append([f'TEMPLATE_{template_num}_CLOSING', closing])

                    if i < 2:  # Add empty row between templates (not after last one)
                        tab_data.append(['', ''])

        # QR Codes
        tab_data.append(['', ''])
        qr_videos = select_qr_code_videos(bucket_path, bucket_name)

        # Generate QR code PNGs
        qr_output_dir = os.path.join(base_path, 'qr_codes')

        top_qr_filename = f"{args.hashtag}_{bucket_name}_top.png"
        top_qr_path = os.path.join(qr_output_dir, top_qr_filename)
        generate_qr_codes(qr_videos['top_performer']['url'], top_qr_path)
        qr_codes_generated += 1

        bottom_qr_filename = f"{args.hashtag}_{bucket_name}_bottom.png"
        bottom_qr_path = os.path.join(qr_output_dir, bottom_qr_filename)
        generate_qr_codes(qr_videos['bottom_performer']['url'], bottom_qr_path)
        qr_codes_generated += 1

        tab_data.append(['QR_CODE_TOP_FILE', top_qr_filename])
        tab_data.append(['QR_CODE_TOP_URL', qr_videos['top_performer']['url']])
        tab_data.append(['QR_CODE_TOP_VIEWS', format_views(qr_videos['top_performer']['views'])])
        tab_data.append(['', ''])
        tab_data.append(['QR_CODE_BOTTOM_FILE', bottom_qr_filename])
        tab_data.append(['QR_CODE_BOTTOM_URL', qr_videos['bottom_performer']['url']])
        tab_data.append(['QR_CODE_BOTTOM_VIEWS', format_views(qr_videos['bottom_performer']['views'])])

        # Convert to DataFrame
        df = pd.DataFrame(tab_data, columns=['Field Name', 'Value'])
        all_tabs[f'Formula_{bucket_name}'] = df

    print(f"Generating {qr_codes_generated} QR codes...")

    # =============================
    # STEP 4: Write Excel File
    # =============================
    excel_filename = f"{args.hashtag}_creator_data.xlsx"
    excel_path = os.path.join(base_path, excel_filename)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for tab_name, df in all_tabs.items():
            df.to_excel(writer, sheet_name=tab_name, index=False)

    # =============================
    # STEP 5: Print Success Message
    # =============================
    print(f"\n✓ Extraction complete")
    print(f"  Excel: {excel_path}")
    print(f"  QR codes: {qr_codes_generated} generated in qr_codes/")


if __name__ == '__main__':
    main()
```

---

### Implementation Notes for LLM

**TODO items in skeleton above**:
1. Implement `aggregate_content_classifications()` - see Section 3.2 for full function
2. Extract caption analysis fields from aggregated Stage 2.7 data

**Testing checklist**:
- [ ] Script runs without errors
- [ ] Excel file created with 3 tabs
- [ ] Each tab has ~63 rows (fields + empty rows + dividers)
- [ ] 6 QR code PNGs generated in qr_codes/ subdirectory
- [ ] QR codes scan successfully and open TikTok videos
- [ ] Field values are not placeholders (actual data extracted)

**Error handling**:
Script should exit with clear error if:
- `winner_analysis.json` not found
- JSON files malformed
- Missing required fields (e.g., `top_3_buckets` array empty)
- Cannot write Excel file (permissions issue)
- Cannot create qr_codes/ directory

---

**END OF SECTION 3.1**

This section is complete and self-contained for LLM implementation of `extract_creator_data.py`.


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
  Total fields: 62
```

---

### Complete Field List

**Excel Structure**: Single tab with two-column format (Field Name | Value)

**Total Fields**: ~62 fields across 3 pages

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

    ('KEYWORD_1', 'gut health'),  # Top 4 keywords
    ('KEYWORD_2', 'protein'),
    ('KEYWORD_3', 'anti-inflammatory'),
    ('KEYWORD_4', 'metabolism'),
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

    # --- Section 4: Quantitative Intelligence ---
    # Note: 9 formulas = 3 winning buckets × 3 formulas per bucket
    # Bucket names + formula names extracted via extract_formula_names_per_bucket()
    ('FORMULA_COUNT', '9'),  # Total formulas delivered
    ('', ''),

    # Bucket 1: Duration + 3 formulas
    ('BUCKET_1_NAME', '18-33s'),  # From winner_analysis.json → top_3_buckets[0]
    ('BUCKET_1_FORMULA_1_NAME', 'The Silent-to-Vocal Engagement Journey'),  # From ml_analysis/llm/winning_formulas.json
    ('BUCKET_1_FORMULA_2_NAME', 'The Visual Storytelling Formula'),
    ('BUCKET_1_FORMULA_3_NAME', 'The Vocal Variety Formula'),
    ('', ''),

    # Bucket 2: Duration + 3 formulas
    ('BUCKET_2_NAME', '13-18s'),  # From winner_analysis.json → top_3_buckets[1]
    ('BUCKET_2_FORMULA_1_NAME', 'The Transformation Story'),
    ('BUCKET_2_FORMULA_2_NAME', 'The Ingredient Deep-Dive'),
    ('BUCKET_2_FORMULA_3_NAME', 'The Side-by-Side Comparison'),
    ('', ''),

    # Bucket 3: Duration + 3 formulas
    ('BUCKET_3_NAME', '60-90s'),  # From winner_analysis.json → top_3_buckets[2]
    ('BUCKET_3_FORMULA_1_NAME', 'The Step-by-Step Tutorial'),
    ('BUCKET_3_FORMULA_2_NAME', 'The Expert Interview Format'),
    ('BUCKET_3_FORMULA_3_NAME', 'The Before-After Journey'),
]
```

**Notes**:
- Total fields: ~125 (including section dividers and empty rows)
- Data fields: ~62 (excluding dividers and empty rows)
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

**Purpose**: Aggregate content patterns from per-bucket validated files

**Used by**: Report 1 (all-buckets aggregation for Content Intelligence section)

**Implementation**: See Section 0.5.1 for complete function definition

**Usage in Report 1 (All-Buckets Aggregation)**:
```python
# Aggregate across ALL winning buckets for market-level insights
from collections import Counter

# Combine aggregations from all buckets
combined = {
    'content_category': Counter(),
    'hook_strategy': Counter(),
    'pain_points': Counter(),
    'keywords': Counter(),
    'engagement_drivers': Counter(),
    'content_tactics': Counter(),
    'caption_hook_type': Counter(),
    'caption_cta_type': Counter(),
    'hashtag_counts': []
}

for bucket_name in winning_buckets:  # e.g., ['18-33s', '33-60s', '60-90s']
    # Get bucket-specific aggregation
    bucket_data = aggregate_content_classifications(
        bucket_name=bucket_name,
        base_path=base_path,
        performer_type="top"
    )

    if bucket_data:
        # Combine Counters from this bucket
        combined['content_category'].update(bucket_data['content_category'])
        combined['hook_strategy'].update(bucket_data['hook_strategy'])
        combined['pain_points'].update(bucket_data['pain_points'])
        combined['keywords'].update(bucket_data['keywords'])
        combined['engagement_drivers'].update(bucket_data['engagement_drivers'])
        combined['content_tactics'].update(bucket_data['content_tactics'])
        combined['caption_hook_type'].update(bucket_data['caption_hook_type'])
        combined['caption_cta_type'].update(bucket_data['caption_cta_type'])
        combined['hashtag_counts'].extend(bucket_data.get('hashtag_counts', []))

# Extract top N from combined data (ALL 300 videos across all buckets)
top_3_categories = [c for c, _ in combined['content_category'].most_common(3)]
# Returns: ['recipe_tutorial', 'wellness_practice', 'supplement_review']

top_5_keywords = [k for k, _ in combined['keywords'].most_common(5)]
# Returns: ['gut_health', 'protein', 'fiber', 'supplements', 'anti-inflammatory']
```

**Data Source**: `content_analysis/validated/bucket_{name}/*_content.json` (per-bucket organization)

---

#### Function 4: `extract_formula_names_per_bucket()`

**Purpose**: Extract Stage 7 LLM-generated formula names for Page 3 Section 4 (Quantitative Intelligence)

**When to Use**: Report 1 only - extracts 9 formula names (3 per winning bucket) for the creative reports summary

**Input Parameters**:
- `analysis_path` (str): Path to analysis directory
  - Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/`
- `winning_buckets` (list): List of 3 winning bucket names
  - Example: `["18-33s", "13-18s", "60-90s"]`
  - Source: `winner_analysis.json` → `top_3_buckets`

**Process**:
1. Loop through each winning bucket
2. For each bucket:
   - Build path to `ml_analysis/llm/winning_formulas.json`
   - Load JSON file
   - Extract `creative_reports[0-2].formula_name` (3 formulas per bucket)
3. Return dict with bucket names and formula arrays

**Implementation**:
```python
def extract_formula_names_per_bucket(analysis_path, winning_buckets):
    """
    Extract 3 formula names per winning bucket (9 total) from Stage 7 output.

    Args:
        analysis_path: Path to analysis directory
        winning_buckets: List of 3 bucket names from winner_analysis.json

    Returns:
        dict: {
            "bucket_names": ["18-33s", "13-18s", "60-90s"],
            "bucket_formulas": {
                "bucket_1": ["Formula 1", "Formula 2", "Formula 3"],
                "bucket_2": ["Formula 4", "Formula 5", "Formula 6"],
                "bucket_3": ["Formula 7", "Formula 8", "Formula 9"]
            }
        }
    """
    import json

    bucket_formulas = {}

    for idx, bucket_name in enumerate(winning_buckets, start=1):
        bucket_path = f"{analysis_path}/buckets/bucket_{bucket_name}"
        winning_formulas_path = f"{bucket_path}/ml_analysis/llm/winning_formulas.json"

        with open(winning_formulas_path, 'r') as f:
            winning_formulas = json.load(f)

        # Extract 3 formula names from creative_reports array
        formulas = [
            report["formula_name"]
            for report in winning_formulas["creative_reports"][:3]
        ]

        bucket_formulas[f"bucket_{idx}"] = formulas

    return {
        "bucket_names": winning_buckets,
        "bucket_formulas": bucket_formulas
    }
```

**Output Format**:
```python
{
    "bucket_names": ["18-33s", "13-18s", "60-90s"],
    "bucket_formulas": {
        "bucket_1": [
            "The Silent-to-Vocal Engagement Journey",
            "The Visual Storytelling Formula",
            "The Vocal Variety Formula"
        ],
        "bucket_2": [
            "The Transformation Story",
            "The Ingredient Deep-Dive",
            "The Side-by-Side Comparison"
        ],
        "bucket_3": [
            "The Step-by-Step Tutorial",
            "The Expert Interview Format",
            "The Before-After Journey"
        ]
    }
}
```

**Usage in Main Script**:
```python
# Extract formula data
formula_data = extract_formula_names_per_bucket(analysis_path, winning_buckets)

# Map to Excel fields
data['BUCKET_1_NAME'] = formula_data['bucket_names'][0]
data['BUCKET_1_FORMULA_1_NAME'] = formula_data['bucket_formulas']['bucket_1'][0]
data['BUCKET_1_FORMULA_2_NAME'] = formula_data['bucket_formulas']['bucket_1'][1]
data['BUCKET_1_FORMULA_3_NAME'] = formula_data['bucket_formulas']['bucket_1'][2]

data['BUCKET_2_NAME'] = formula_data['bucket_names'][1]
data['BUCKET_2_FORMULA_1_NAME'] = formula_data['bucket_formulas']['bucket_2'][0]
# ... and so on for all 12 fields
```

**Data Source**: `{bucket_path}/ml_analysis/llm/winning_formulas.json` → `creative_reports[].formula_name`

**Validation Status**: ✅ **Verified** with rollo_test4/wellness_test4 data

---

#### Function 5: Inline Calculations

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

#### File 5: `content_analysis/validated/bucket_{name}/{video_id}_content.json`

**Location**: `/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/content_analysis/validated/bucket_{bucket_name}/{video_id}_content.json` (per-bucket organization)

**Purpose**: Stage 2.7 LLM content classifications per video (organized by bucket, includes performer_type field)

**Structure**:
```json
{
  "video_id": "7540717847325003039",
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
  },
  "taxonomy_version": "stage2.6_output",
  "confidence": "high",
  "transcript_available": true,
  "bucket": "18-33s",
  "performer_type": "top"
}
```

**Fields Used**:
- ALL fields for aggregation via `aggregate_content_classifications()`
- `performer_type` → Filter by "top" or "bottom"
- `bucket` → Self-documenting bucket identifier
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
        bucket_aggregated = aggregate_content_classifications(
            bucket_name=bucket_name,
            base_path=base_path,
            performer_type="top"
        )

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
    # STEP 6: Extract Formula Names from Stage 7 (Function 4)
    # =============================
    # Extract bucket names and formula names for Page 3 Section 4
    formula_data = extract_formula_names_per_bucket(base_path, winning_buckets)

    # formula_data contains:
    # - bucket_names: ["18-33s", "13-18s", "60-90s"]
    # - bucket_formulas: {
    #     "bucket_1": ["Formula 1", "Formula 2", "Formula 3"],
    #     "bucket_2": ["Formula 4", "Formula 5", "Formula 6"],
    #     "bucket_3": ["Formula 7", "Formula 8", "Formula 9"]
    #   }

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
        tab_data.append([f'KEYWORD_{i}', keyword])

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

    # Section 4: Quantitative Intelligence
    tab_data.append(['FORMULA_COUNT', '9'])
    tab_data.append(['', ''])

    # 12 fields: 3 bucket names + 9 formula names (3 per bucket)
    for bucket_idx in range(1, 4):
        bucket_key = f'bucket_{bucket_idx}'

        # Add bucket name
        tab_data.append([f'BUCKET_{bucket_idx}_NAME', formula_data['bucket_names'][bucket_idx-1]])

        # Add 3 formula names for this bucket
        tab_data.append([f'BUCKET_{bucket_idx}_FORMULA_1_NAME', formula_data['bucket_formulas'][bucket_key][0]])
        tab_data.append([f'BUCKET_{bucket_idx}_FORMULA_2_NAME', formula_data['bucket_formulas'][bucket_key][1]])
        tab_data.append([f'BUCKET_{bucket_idx}_FORMULA_3_NAME', formula_data['bucket_formulas'][bucket_key][2]])
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

---

## Section 3.3: `extract_competitor_data.py` - COMPLETE IMPLEMENTATION GUIDE

### Overview

**Purpose**: Extract single competitor deep dive analysis data for client executive report (Report 3)

**Report Type**: Report 3 from Stage8MVP_Reports.md Section 3

**Deliverable**: 1 Excel file + 1 QR code for competitor's top video

**CLI Usage**:
```bash
python extract_competitor_data.py --client acme --competitor drinkpoppi --mode top --strategy contrastive
```

**Output Files**:
```
/data/clients/acme/competitors/drinkpoppi/top_contrastive/
├── drinkpoppi_analysis_data.xlsx (single tab with all pages)
└── qr_codes/
    └── drinkpoppi_top.png
```

**Console Output Pattern**:
```bash
$ python extract_competitor_data.py --client acme --competitor drinkpoppi

Running extraction for competitor: @drinkpoppi
Analyzing 127 videos across 3 winning buckets...
Calculating performance metrics...
Aggregating content intelligence from Stage 2.7 classifications...
Extracting hashtag and mention analysis...
Generating 1 QR code...

✓ Extraction complete
  Excel: /data/clients/acme/competitors/drinkpoppi/top_contrastive/drinkpoppi_analysis_data.xlsx
  QR code: /data/clients/acme/competitors/drinkpoppi/top_contrastive/qr_codes/drinkpoppi_top.png
  Total fields: 68
```

---

### Complete Field List

**Excel Structure**: Single tab with two-column format (Field Name | Value)

**Total Fields**: ~68 fields across 3 pages

```python
# Field structure - two-column format: Field Name | Value
fields = [
    # =============================
    # PAGE 1: EXECUTIVE OVERVIEW
    # =============================
    ('PAGE_1_EXECUTIVE_OVERVIEW', ''),  # Section divider
    ('', ''),  # Empty row

    # --- Header Section ---
    ('COMPETITOR_HANDLE', '@drinkpoppi'),  # From config or CLI parameter
    ('ANALYSIS_PERIOD', 'Last 90 days'),  # Static
    ('VIDEOS_ANALYZED', '127'),  # Sum of selected_count from winning buckets

    # =============================
    # PAGE 2: CONTENT STRATEGY ANALYSIS
    # =============================
    ('', ''),
    ('PAGE_2_CONTENT_STRATEGY', ''),  # Section divider
    ('', ''),

    # --- Section 1: Duration Distribution ---
    ('BUCKET_0_3S_PCT', '3'),  # From winner_analysis.json → bucket_distribution
    ('BUCKET_3_9S_PCT', '8'),
    ('BUCKET_9_13S_PCT', '12'),
    ('BUCKET_13_18S_PCT', '18'),
    ('BUCKET_18_33S_PCT', '32'),
    ('BUCKET_33_60S_PCT', '22'),
    ('BUCKET_60_90S_PCT', '4'),
    ('BUCKET_90_120S_PCT', '1'),
    ('', ''),
    ('PRIMARY_FOCUS_BUCKET', '18-33s'),  # Bucket with highest %

    # --- Section 2: Performance by Duration ---
    ('', ''),
    # Top 3 performing buckets (sorted by composite score: engagement primary, views secondary)
    ('PERF_BUCKET_1_NAME', '18-33s'),
    ('PERF_BUCKET_1_AVG_VIEWS', '620K'),
    ('PERF_BUCKET_1_AVG_ENG', '1.5'),
    ('PERF_BUCKET_1_STARS', '⭐⭐⭐⭐⭐'),
    ('PERF_BUCKET_1_IS_SWEET_SPOT', 'True'),  # Only rank 1 = True
    ('', ''),

    ('PERF_BUCKET_2_NAME', '13-18s'),
    ('PERF_BUCKET_2_AVG_VIEWS', '580K'),
    ('PERF_BUCKET_2_AVG_ENG', '1.3'),
    ('PERF_BUCKET_2_STARS', '⭐⭐⭐⭐'),
    ('PERF_BUCKET_2_IS_SWEET_SPOT', 'False'),
    ('', ''),

    ('PERF_BUCKET_3_NAME', '33-60s'),
    ('PERF_BUCKET_3_AVG_VIEWS', '490K'),
    ('PERF_BUCKET_3_AVG_ENG', '1.4'),
    ('PERF_BUCKET_3_STARS', '⭐⭐⭐⭐'),
    ('PERF_BUCKET_3_IS_SWEET_SPOT', 'False'),
    ('', ''),

    ('SWEET_SPOT_BUCKET', '18-33s'),  # Extracted from rank 1
    ('COVERAGE_PERCENTAGE', '72'),  # Top 3 as % of all content

    # --- Section 3: Posting Frequency ---
    ('', ''),
    ('POSTING_FREQUENCY', '14'),  # Videos per week

    # =============================
    # PAGE 3: CREATIVE INTELLIGENCE
    # =============================
    ('', ''),
    ('PAGE_3_CREATIVE_INTELLIGENCE', ''),  # Section divider
    ('', ''),

    # --- Section 1: Content DNA ---
    ('CONTENT_CATEGORY_1', 'Recipe Tutorial'),  # Top 5 from Stage 2.7
    ('CONTENT_CATEGORY_1_PCT', '38'),
    ('CONTENT_CATEGORY_1_DESC', 'Step-by-step cooking instructions'),
    ('CONTENT_CATEGORY_2', 'Wellness Practice'),
    ('CONTENT_CATEGORY_2_PCT', '28'),
    ('CONTENT_CATEGORY_2_DESC', 'Daily health routines and habits'),
    ('CONTENT_CATEGORY_3', 'Supplement Review'),
    ('CONTENT_CATEGORY_3_PCT', '17'),
    ('CONTENT_CATEGORY_3_DESC', 'Product recommendations and reviews'),
    ('CONTENT_CATEGORY_4', 'Expert Interview'),
    ('CONTENT_CATEGORY_4_PCT', '12'),
    ('CONTENT_CATEGORY_4_DESC', 'Professional perspectives'),
    ('CONTENT_CATEGORY_5', 'Personal Testimony'),
    ('CONTENT_CATEGORY_5_PCT', '5'),
    ('CONTENT_CATEGORY_5_DESC', 'Personal success stories'),
    ('', ''),

    ('ENGAGEMENT_DRIVER_1', 'Before/After Reveal'),  # Top 4 from Stage 2.7
    ('ENGAGEMENT_DRIVER_1_PCT', '45'),
    ('ENGAGEMENT_DRIVER_1_DESC', 'Visual transformations'),
    ('ENGAGEMENT_DRIVER_2', 'Specific Metrics'),
    ('ENGAGEMENT_DRIVER_2_PCT', '42'),
    ('ENGAGEMENT_DRIVER_2_DESC', '"Lost 15 lbs in 30 days"'),
    ('ENGAGEMENT_DRIVER_3', 'Personal Testimony'),
    ('ENGAGEMENT_DRIVER_3_PCT', '38'),
    ('ENGAGEMENT_DRIVER_3_DESC', '"This worked for me..."'),
    ('ENGAGEMENT_DRIVER_4', 'Expert Credentials'),
    ('ENGAGEMENT_DRIVER_4_PCT', '28'),
    ('ENGAGEMENT_DRIVER_4_DESC', '"Registered nutritionist here..."'),

    # --- Section 2: Execution Playbook ---
    ('', ''),
    ('HOOK_STRATEGY_1', 'Question Hook'),  # Top 4 from Stage 2.7
    ('HOOK_STRATEGY_1_PCT', '42'),
    ('HOOK_STRATEGY_1_DESC', 'Opens with engaging question'),
    ('HOOK_STRATEGY_2', 'Problem-Solution'),
    ('HOOK_STRATEGY_2_PCT', '31'),
    ('HOOK_STRATEGY_2_DESC', 'Identifies pain point, offers solution'),
    ('HOOK_STRATEGY_3', 'Direct Statement'),
    ('HOOK_STRATEGY_3_PCT', '18'),
    ('HOOK_STRATEGY_3_DESC', 'Bold claim or fact'),
    ('HOOK_STRATEGY_4', 'Curiosity Gap'),
    ('HOOK_STRATEGY_4_PCT', '9'),
    ('HOOK_STRATEGY_4_DESC', 'Creates mystery or intrigue'),
    ('', ''),

    ('CTA_STRATEGY_1', 'Link in Bio'),  # Top 4 from Stage 2.7 caption_cta_type
    ('CTA_STRATEGY_1_PCT', '38'),
    ('CTA_STRATEGY_1_DESC', 'Directs viewers to profile link'),
    ('CTA_STRATEGY_2', 'Follow for More'),
    ('CTA_STRATEGY_2_PCT', '32'),
    ('CTA_STRATEGY_2_DESC', 'Encourages account following'),
    ('CTA_STRATEGY_3', 'Save This Post'),
    ('CTA_STRATEGY_3_PCT', '21'),
    ('CTA_STRATEGY_3_DESC', 'Prompts content bookmarking'),
    ('CTA_STRATEGY_4', 'Tag a Friend'),
    ('CTA_STRATEGY_4_PCT', '9'),
    ('CTA_STRATEGY_4_DESC', 'Drives viral sharing'),
    ('', ''),

    ('PAIN_POINT_1', 'Bloating/Digestive Issues'),  # Top 5 from Stage 2.7
    ('PAIN_POINT_1_PCT', '48'),
    ('PAIN_POINT_2', 'Low Energy/Fatigue'),
    ('PAIN_POINT_2_PCT', '42'),
    ('PAIN_POINT_3', 'Weight Management'),
    ('PAIN_POINT_3_PCT', '38'),
    ('PAIN_POINT_4', 'Inflammation'),
    ('PAIN_POINT_4_PCT', '32'),
    ('PAIN_POINT_5', 'Gut Health'),
    ('PAIN_POINT_5_PCT', '28'),
    ('', ''),

    ('KEYWORD_1', 'gut health'),  # Top 5 from Stage 2.7
    ('KEYWORD_2', 'protein'),
    ('KEYWORD_3', 'anti-inflammatory'),
    ('KEYWORD_4', 'metabolism'),
    ('KEYWORD_5', 'fiber'),
    ('', ''),

    ('CONTENT_TACTIC_1', 'Direct-to-Camera'),  # Top 4 from Stage 2.7
    ('CONTENT_TACTIC_1_PCT', '52'),
    ('CONTENT_TACTIC_2', 'Voiceover + B-roll'),
    ('CONTENT_TACTIC_2_PCT', '31'),
    ('CONTENT_TACTIC_3', 'Text-Heavy Overlays'),
    ('CONTENT_TACTIC_3_PCT', '24'),
    ('CONTENT_TACTIC_4', 'Product Demonstration'),
    ('CONTENT_TACTIC_4_PCT', '18'),

    # --- Section 3: Hashtag Strategy ---
    ('', ''),
    ('HASHTAG_1', '#nutrition'),  # Top 10 hashtags
    ('HASHTAG_1_PCT', '82'),
    ('HASHTAG_2', '#healthylifestyle'),
    ('HASHTAG_2_PCT', '68'),
    ('HASHTAG_3', '#wellness'),
    ('HASHTAG_3_PCT', '54'),
    ('HASHTAG_4', '#guthealth'),
    ('HASHTAG_4_PCT', '47'),
    ('HASHTAG_5', '#protein'),
    ('HASHTAG_5_PCT', '43'),
    ('HASHTAG_6', '#healthyeating'),
    ('HASHTAG_6_PCT', '38'),
    ('HASHTAG_7', '#fitfood'),
    ('HASHTAG_7_PCT', '32'),
    ('HASHTAG_8', '#cleaneating'),
    ('HASHTAG_8_PCT', '28'),
    ('HASHTAG_9', '#nutritionist'),
    ('HASHTAG_9_PCT', '24'),
    ('HASHTAG_10', '#healthyliving'),
    ('HASHTAG_10_PCT', '21'),
    ('', ''),

    ('TOTAL_UNIQUE_HASHTAGS', '28'),
    ('AVG_HASHTAGS_PER_VIDEO', '9'),
    ('STRATEGY_TYPE', 'Diversified'),  # "Diversified" if >20, else "Focused"

    # --- Section 4: Caption Strategy ---
    ('', ''),
    ('AVG_HASHTAG_COUNT', '12'),
    ('TOP_CTA_TYPE', 'Follow me'),
    ('TOP_CTA_TYPE_PCT', '52'),

    # --- Section 5: Content Sourcing Strategy ---
    ('', ''),
    ('ORIGINAL_CONTENT_PCT', '58'),  # 100 - repost_rate
    ('REPOSTED_AFFILIATE_PCT', '42'),  # From extract_mention_analysis()
    ('', ''),

    ('AFFILIATE_1_HANDLE', '@fitnessguru123'),  # Top 5 affiliates
    ('AFFILIATE_1_PCT', '18'),
    ('AFFILIATE_1_COUNT', '54'),
    ('AFFILIATE_2_HANDLE', '@healthcoach_jane'),
    ('AFFILIATE_2_PCT', '12'),
    ('AFFILIATE_2_COUNT', '36'),
    ('AFFILIATE_3_HANDLE', '@nutritionpro'),
    ('AFFILIATE_3_PCT', '8'),
    ('AFFILIATE_3_COUNT', '24'),
    ('AFFILIATE_4_HANDLE', '@wellnesswarrior'),
    ('AFFILIATE_4_PCT', '5'),
    ('AFFILIATE_4_COUNT', '15'),
    ('AFFILIATE_5_HANDLE', '@cleaneatingclub'),
    ('AFFILIATE_5_PCT', '4'),
    ('AFFILIATE_5_COUNT', '12'),
    ('', ''),

    ('TOTAL_UNIQUE_MENTIONS', '47'),

    # --- Section 6: Creative Formulas ---
    ('', ''),
    # Bucket 1 (First winning bucket)
    ('BUCKET_1_NAME', '18-33s'),  # From winner_analysis.json -> top_3_buckets[0]
    ('BUCKET_1_FORMULA_1_NAME', 'The Silent-to-Vocal Engagement Journey'),  # From ml_analysis/llm/winning_formulas.json
    ('BUCKET_1_FORMULA_2_NAME', 'The Visual Storytelling Formula'),
    ('BUCKET_1_FORMULA_3_NAME', 'The Vocal Variety Formula'),
    ('', ''),
    # Bucket 2 (Second winning bucket)
    ('BUCKET_2_NAME', '13-18s'),  # From winner_analysis.json -> top_3_buckets[1]
    ('BUCKET_2_FORMULA_1_NAME', 'The Transformation Story'),
    ('BUCKET_2_FORMULA_2_NAME', 'The Personal Journey'),
    ('BUCKET_2_FORMULA_3_NAME', 'The Quick Win'),
    ('', ''),
    # Bucket 3 (Third winning bucket)
    ('BUCKET_3_NAME', '60-90s'),  # From winner_analysis.json -> top_3_buckets[2]
    ('BUCKET_3_FORMULA_1_NAME', 'The Step-by-Step Tutorial'),
    ('BUCKET_3_FORMULA_2_NAME', 'The Expert Breakdown'),
    ('BUCKET_3_FORMULA_3_NAME', 'The Deep Dive'),

    # --- QR Code Metadata ---
    ('', ''),
    ('QR_CODE_FILE', 'drinkpoppi_top.png'),
    ('QR_CODE_URL', 'https://www.tiktok.com/@drinkpoppi/video/7540717847325003039'),
    ('QR_CODE_VIEWS', '620K'),
    ('QR_CODE_ENGAGEMENT', '1.5'),
    ('QR_CODE_DURATION', '22s'),
    ('QR_CODE_BUCKET', '18-33s'),
]
```

**Notes**:
- Total fields: ~140 (including section dividers and empty rows)
- Field naming: `UPPERCASE_WITH_UNDERSCORES`
- Multi-value fields use numbered suffixes
- Empty rows for visual separation
- Section dividers use equals signs

---

### Required Functions

This section defines all functions needed for `extract_competitor_data.py`. Functions are documented inline for self-contained implementation.

---

#### Function 1: `calculate_engagement_metrics()`

**Purpose**: Calculate real engagement rate from TikTok video metadata

**Used by**: Report 3 (this script), all reports

**Input**: Video metadata dictionary with engagement fields

**Output**: Engagement rate as float (percentage)

**Implementation**:
```python
def calculate_engagement_metrics(video_metadata):
    """
    Calculate engagement rate from TikTok video metadata.

    Formula: (likes + comments + shares + saves) / views × 100

    Input fields (from selected_videos.json):
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

---

#### Function 2: `calculate_bucket_distribution()`

**Purpose**: Calculate percentage of videos in each of 8 duration buckets

**Used by**: Report 3 (Duration Distribution section)

**Input**: Path to winner_analysis.json

**Output**: Dict mapping bucket name → percentage

**Implementation**:
```python
def calculate_bucket_distribution(winner_analysis_path):
    """
    Calculate percentage distribution across all 8 duration buckets.

    Args:
        winner_analysis_path: Path to winner_analysis.json file

    Returns:
        dict: Bucket name → percentage (rounded to integer)

    Example:
        {
            "0-3s": 3,
            "3-9s": 8,
            "9-13s": 12,
            "13-18s": 18,
            "18-33s": 32,
            "33-60s": 22,
            "60-90s": 4,
            "90-120s": 1
        }
    """
    import json

    with open(winner_analysis_path) as f:
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

**Data Source**: `winner_analysis.json → bucket_distribution`

---

#### Function 3: `rank_competitor_top_buckets()`

**Purpose**: Rank a single competitor's top 3 buckets by performance and assign star ratings

**Used by**: Report 3 (Performance by Duration section)

**Input**: Client ID and competitor handle

**Output**: List of dicts with bucket rankings

**Implementation**:
```python
def rank_competitor_top_buckets(client_id, competitor_handle):
    """
    Rank competitor's top 3 buckets by performance.

    Ranking Criteria:
    1. Primary: Average engagement rate (higher is better)
    2. Secondary: Average views (higher is better)

    Args:
        client_id: Client identifier
        competitor_handle: Competitor handle without @ symbol

    Returns:
        List of dicts with rankings, sorted by performance DESC

    Example:
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
    """
    import json
    import os

    # Discover analysis directory
    base_path = f"/data/clients/{client_id}/competitors/{competitor_handle}"
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
        # Load selected videos for this bucket
        bucket_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"

        with open(bucket_path) as f:
            selected_data = json.load(f)

        top_count = selected_data["top_count"]
        top_videos = selected_data["videos"][:top_count]

        # Calculate avg views
        avg_views = sum(v["playCount"] for v in top_videos) / len(top_videos)

        # Calculate avg engagement
        engagement_rates = []
        for video in top_videos:
            engagement = calculate_engagement_metrics(video)
            engagement_rates.append(engagement)

        avg_engagement = sum(engagement_rates) / len(engagement_rates)

        bucket_data.append({
            "bucket": bucket,
            "avg_views": int(avg_views),
            "avg_engagement": round(avg_engagement, 1)
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

**Data Source**: `selected_videos.json` per bucket

---

#### Function 4: `aggregate_content_classifications()`

**Purpose**: Aggregate 120 individual Stage 2.7 classifications into frequency distributions

**Used by**: Report 3 (Content DNA and Execution Playbook sections)

**Implementation**: Same as Report 1 - see Section 3.2 Function 3

---

#### Function 5: `extract_hashtag_analysis()`

**Purpose**: Extract hashtag patterns from all winning buckets

**Used by**: Report 3 (Hashtag Strategy section)

**Input**: Client ID and competitor handle

**Output**: Dict with hashtag statistics

**Implementation**:
```python
def extract_hashtag_analysis(client_id, competitor_handle):
    """
    Extract hashtag usage patterns from competitor's winning buckets.

    Args:
        client_id: Client identifier
        competitor_handle: Competitor handle without @ symbol

    Returns:
        dict: {
            "total_unique_hashtags": 28,
            "avg_hashtags_per_video": 9.2,
            "top_5_concentration": 65,
            "top_10_hashtags": [
                {"tag": "#nutrition", "usage_pct": 82, "video_count": 104},
                {"tag": "#healthylifestyle", "usage_pct": 68, "video_count": 86},
                ...
            ]
        }
    """
    import json
    import os
    from collections import Counter

    # Discover analysis directory
    base_path = f"/data/clients/{client_id}/competitors/{competitor_handle}"
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(f"No analysis directory found for {competitor_handle}")

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]

    # Collect hashtags from all winning buckets
    all_hashtags = []
    total_videos = 0

    for bucket in winning_buckets:
        bucket_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"

        with open(bucket_path) as f:
            data = json.load(f)

        top_count = data["top_count"]
        top_videos = data["videos"][:top_count]
        total_videos += len(top_videos)

        for video in top_videos:
            hashtags = video.get("hashtags", [])
            for hashtag in hashtags:
                tag_name = hashtag.get("name", "")
                if tag_name:
                    # Normalize to lowercase to avoid duplicates (e.g., "GymTok" vs "gymtok")
                    all_hashtags.append(tag_name.lower())

    # Calculate statistics
    unique_hashtags = set(all_hashtags)
    hashtag_counter = Counter(all_hashtags)

    total_hashtags = len(all_hashtags)
    avg_hashtags_per_video = round(total_hashtags / total_videos, 1) if total_videos > 0 else 0

    # Top 10 hashtags
    top_10 = []
    for tag, count in hashtag_counter.most_common(10):
        usage_pct = round((count / total_videos) * 100)
        top_10.append({
            "tag": f"#{tag}",
            "usage_pct": usage_pct,
            "video_count": count
        })

    # Top 5 concentration
    top_5_count = sum(count for _, count in hashtag_counter.most_common(5))
    top_5_concentration = round((top_5_count / total_hashtags) * 100) if total_hashtags > 0 else 0

    return {
        "total_unique_hashtags": len(unique_hashtags),
        "avg_hashtags_per_video": avg_hashtags_per_video,
        "top_5_concentration": top_5_concentration,
        "top_10_hashtags": top_10
    }
```

**Data Source**: `selected_videos.json → videos[].hashtags[]`

---

#### Function 6: `extract_mention_analysis()`

**Purpose**: Extract @mention patterns for content sourcing analysis

**Used by**: Report 3 (Content Sourcing Strategy section)

**Input**: Client ID and competitor handle

**Output**: Dict with mention statistics

**Implementation**:
```python
def extract_mention_analysis(client_id, competitor_handle):
    """
    Extract @mention patterns from video captions.

    Detects:
    - Videos with @mentions (potential UGC/affiliate content)
    - Repost indicators ("repost", "via", "credit", "by", "from")

    Args:
        client_id: Client identifier
        competitor_handle: Competitor handle without @ symbol

    Returns:
        dict: {
            "total_videos": 127,
            "videos_with_mentions": 53,
            "mention_rate": 42,
            "repost_rate": 42,
            "total_unique_mentions": 47,
            "top_10_mentions": [
                {"handle": "@fitnessguru123", "percentage": 18, "mention_count": 23},
                ...
            ]
        }
    """
    import json
    import os
    import re
    from collections import Counter

    # Discover analysis directory
    base_path = f"/data/clients/{client_id}/competitors/{competitor_handle}"
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(f"No analysis directory found for {competitor_handle}")

    analysis_dir = analysis_dirs[0]
    competitor_path = f"{base_path}/{analysis_dir}"

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]

    # Collect mentions from all winning buckets
    all_mentions = []
    videos_with_mentions = 0
    total_videos = 0

    repost_indicators = ["repost", "via", "credit", "by", "from"]

    for bucket in winning_buckets:
        bucket_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"

        with open(bucket_path) as f:
            data = json.load(f)

        top_count = data["top_count"]
        top_videos = data["videos"][:top_count]
        total_videos += len(top_videos)

        for video in top_videos:
            caption = video.get("text", "").lower()

            # Extract @mentions
            mentions = re.findall(r'@(\w+)', caption)

            # Check repost indicators
            has_repost_indicator = any(indicator in caption for indicator in repost_indicators)

            if mentions or has_repost_indicator:
                videos_with_mentions += 1
                all_mentions.extend(mentions)

    # Calculate statistics
    unique_mentions = set(all_mentions)
    mention_counter = Counter(all_mentions)

    mention_rate = round((videos_with_mentions / total_videos) * 100) if total_videos > 0 else 0
    repost_rate = mention_rate  # Same metric for this analysis

    # Top 10 mentions
    top_10 = []
    for handle, count in mention_counter.most_common(10):
        percentage = round((count / total_videos) * 100, 1)
        top_10.append({
            "handle": f"@{handle}",
            "percentage": percentage,
            "mention_count": count
        })

    return {
        "total_videos": total_videos,
        "videos_with_mentions": videos_with_mentions,
        "mention_rate": mention_rate,
        "repost_rate": repost_rate,
        "total_unique_mentions": len(unique_mentions),
        "top_10_mentions": top_10
    }
```

**Data Source**: `selected_videos.json → videos[].text`

---

#### Function 7: `select_qr_code_videos()`

**Purpose**: Select top/bottom performer videos for QR code generation

**Used by**: Report 3 (1 QR code for top performer)

**Input**: Bucket path and performance group

**Output**: Dict with video metadata

**Implementation**:
```python
def select_qr_code_videos(bucket_path, performance_group="top"):
    """
    Select top or bottom performer video for QR code generation.

    Args:
        bucket_path: Path to bucket folder
        performance_group: "top" or "bottom"

    Returns:
        dict: {
            "video_id": "7540717847325003039",
            "url": "https://www.tiktok.com/@user/video/7540717847325003039",
            "views": 6700000,
            "engagement": 2.0,
            "duration": 21
        }
    """
    import json

    with open(f"{bucket_path}/selected_videos.json") as f:
        data = json.load(f)

    if performance_group == "top":
        # First video in array = highest views
        video = data["videos"][0]
    elif performance_group == "bottom":
        # Last video among bottom performers
        top_count = data["top_count"]
        bottom_count = data["bottom_count"]
        video = data["videos"][top_count + bottom_count - 1]
    else:
        raise ValueError(f"Invalid performance_group: {performance_group}")

    engagement = calculate_engagement_metrics(video)

    return {
        "video_id": video["id"],
        "url": video["webVideoUrl"],
        "views": video["playCount"],
        "engagement": engagement,
        "duration": video["duration"]
    }
```

**Data Source**: `selected_videos.json`

---

#### Function 8: `generate_qr_codes()`

**Purpose**: Generate QR code PNG files from TikTok URLs

**Used by**: Report 3 (1 QR code)

**Input**: List of video data with URLs

**Output**: None (writes PNG files to disk)

**Implementation**:
```python
def generate_qr_codes(qr_data_list, output_dir):
    """
    Generate QR code PNG files from TikTok URLs.

    Args:
        qr_data_list: List of dicts with "filename" and "url" keys
        output_dir: Directory to save QR code images

    Example:
        qr_data_list = [
            {"filename": "drinkpoppi_top.png", "url": "https://www.tiktok.com/@user/video/123"}
        ]
    """
    import qrcode
    import os

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for qr_data in qr_data_list:
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data["url"])
        qr.make(fit=True)

        # Generate image
        img = qr.make_image(fill_color="black", back_color="white")

        # Save to file
        output_path = os.path.join(output_dir, qr_data["filename"])
        img.save(output_path)

    print(f"Generated {len(qr_data_list)} QR code(s) in {output_dir}")
```

**Dependencies**: `pip install qrcode[pil]`

---

#### Function 9: Inline Helper Functions

These are simple formatting/calculation helpers:

```python
def format_views(view_count):
    """Format view count with K or M suffix."""
    if view_count >= 1000000:
        return f"{view_count / 1000000:.1f}M"
    elif view_count >= 1000:
        return f"{int(view_count / 1000)}K"
    else:
        return str(view_count)


def calculate_posting_frequency(client_id, competitor_handle):
    """Calculate videos per week from winner_analysis.json."""
    import json
    import os

    base_path = f"/data/clients/{client_id}/competitors/{competitor_handle}"
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(f"No analysis directory found")

    analysis_dir = analysis_dirs[0]
    winner_analysis_path = f"{base_path}/{analysis_dir}/winner_analysis.json"

    with open(winner_analysis_path) as f:
        data = json.load(f)

    total_videos = sum(data["top_100_distribution"].values())

    # Assume 90 days analysis period
    weeks = 90 / 7  # ~13 weeks

    posting_freq = round(total_videos / weeks, 1)

    return posting_freq


def determine_hashtag_strategy_type(total_unique_hashtags):
    """Determine if hashtag strategy is Diversified or Focused."""
    return "Diversified" if total_unique_hashtags > 20 else "Focused"


def calculate_original_content_percentage(repost_rate):
    """Calculate original content percentage (inverse of repost rate)."""
    return 100 - int(repost_rate)
```

---

### Data Source File Formats

This section documents the exact JSON structure for all files used by this script.

---

#### File 1: `winner_analysis.json`

**Location**: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/winner_analysis.json`

**Purpose**: Winning buckets and bucket distribution

**Structure**:
```json
{
  "top_3_buckets": ["18-33s", "13-18s", "33-60s"],
  "bucket_distribution": {
    "0-3s": 5,
    "3-9s": 15,
    "9-13s": 22,
    "13-18s": 34,
    "18-33s": 60,
    "33-60s": 41,
    "60-90s": 8,
    "90-120s": 2
  },
  "top_100_distribution": {
    "0-3s": 3,
    "3-9s": 8,
    "9-13s": 12,
    "13-18s": 18,
    "18-33s": 32,
    "33-60s": 22,
    "60-90s": 4,
    "90-120s": 1
  }
}
```

**Fields Used**:
- `top_3_buckets` → Winning bucket names
- `bucket_distribution` → All videos count per bucket (for percentages)
- `top_100_distribution` → Top 100 distribution (for posting frequency)

---

#### File 2: `selected_videos.json` (per bucket)

**Location**: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/buckets/bucket_{name}/selected_videos.json`

**Purpose**: Video metadata for performance calculations

**Structure**: Same as Report 1 - see Section 3.2 File 4

---

#### File 3: `content_analysis/validated/bucket_{name}/{video_id}_content.json`

**Location**: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/content_analysis/validated/bucket_{bucket_name}/{video_id}_content.json` (per-bucket organization)

**Purpose**: Stage 2.7 content classifications (organized by bucket, includes performer_type field)

**Structure**: Same as Report 1 - see Section 3.2 File 5

---

### Complete Implementation Pattern

```python
#!/usr/bin/env python3
"""
extract_competitor_data.py - Report 3: Single Competitor → Client

Generates deep dive competitive intelligence on 1 competitor.

Usage:
    python extract_competitor_data.py --client acme --competitor drinkpoppi --mode top --strategy contrastive
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
    parser = argparse.ArgumentParser(description='Extract Report 3: Single Competitor')
    parser.add_argument('--client', required=True, help='Client ID (e.g., acme)')
    parser.add_argument('--competitor', required=True, help='Competitor handle without @ (e.g., drinkpoppi)')
    parser.add_argument('--mode', default='top', help='Mode (default: top)')
    parser.add_argument('--strategy', default='contrastive', help='Strategy (default: contrastive)')
    args = parser.parse_args()

    print(f"\nRunning extraction for competitor: @{args.competitor}")

    # =============================
    # STEP 2: Build File Paths
    # =============================
    base_path = f"/data/clients/{args.client}/competitors/{args.competitor}"

    # Discover analysis directory
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]
    if not analysis_dirs:
        raise FileNotFoundError(f"No analysis directory found for {args.competitor}")

    analysis_dir = analysis_dirs[0]
    analysis_base_path = f"{base_path}/{analysis_dir}"

    # Create reports/competitor/ directory structure
    reports_base_path = os.path.join(analysis_base_path, 'reports', 'competitor')
    os.makedirs(reports_base_path, exist_ok=True)

    # =============================
    # STEP 3: Load Core Data
    # =============================
    winner_analysis_path = os.path.join(analysis_base_path, 'winner_analysis.json')
    with open(winner_analysis_path) as f:
        winner_data = json.load(f)

    winning_buckets = winner_data['top_3_buckets']

    # Calculate total videos analyzed
    total_videos = 0
    for bucket in winning_buckets:
        bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket}')
        with open(f"{bucket_path}/selected_videos.json") as f:
            data = json.load(f)
        total_videos += data['selected_count']

    print(f"Analyzing {total_videos} videos across {len(winning_buckets)} winning buckets...")

    # =============================
    # STEP 4: Calculate Performance Metrics
    # =============================
    print("Calculating performance metrics...")

    # Bucket distribution
    bucket_percentages = calculate_bucket_distribution(winner_analysis_path)
    primary_focus = max(bucket_percentages, key=bucket_percentages.get)

    # Rank top 3 buckets
    ranked_buckets = rank_competitor_top_buckets(args.client, args.competitor)
    sweet_spot = ranked_buckets[0]['bucket']

    # Coverage percentage
    coverage_pct = sum(bucket_percentages[b] for b in winning_buckets)

    # Posting frequency
    posting_freq = calculate_posting_frequency(args.client, args.competitor)

    # =============================
    # STEP 5: Aggregate Content Intelligence
    # =============================
    print("Aggregating content intelligence from Stage 2.7 classifications...")

    # Aggregate across all winning buckets
    all_content_categories = Counter()
    all_hook_strategies = Counter()
    all_cta_types = Counter()
    all_pain_points = Counter()
    all_keywords = Counter()
    all_engagement_drivers = Counter()
    all_content_tactics = Counter()

    for bucket in winning_buckets:
        aggregated = aggregate_content_classifications(
            bucket_name=bucket,
            base_path=competitor_path,
            performer_type="top"
        )

        if aggregated:
            all_content_categories.update(aggregated['content_category'])
            all_hook_strategies.update(aggregated['hook_strategy'])
            all_cta_types.update(aggregated['caption_cta_type'])
            all_pain_points.update(aggregated['pain_points'])
            all_keywords.update(aggregated['keywords'])
            all_engagement_drivers.update(aggregated['engagement_drivers'])
            all_content_tactics.update(aggregated['content_tactics'])

    # Get top N
    top_5_categories = all_content_categories.most_common(5)
    top_4_hooks = all_hook_strategies.most_common(4)
    top_4_ctas = all_cta_types.most_common(4)
    top_5_pain_points = all_pain_points.most_common(5)
    top_5_keywords = all_keywords.most_common(5)
    top_4_drivers = all_engagement_drivers.most_common(4)
    top_4_tactics = all_content_tactics.most_common(4)

    total_classified = sum(all_content_categories.values())

    # =============================
    # STEP 6: Extract Hashtag & Mention Analysis
    # =============================
    print("Extracting hashtag and mention analysis...")

    hashtag_analysis = extract_hashtag_analysis(args.client, args.competitor)
    mention_analysis = extract_mention_analysis(args.client, args.competitor)

    strategy_type = determine_hashtag_strategy_type(hashtag_analysis['total_unique_hashtags'])
    original_content_pct = calculate_original_content_percentage(mention_analysis['repost_rate'])

    # =============================
    # STEP 7: Select QR Code Video
    # =============================
    print("Generating 1 QR code...")

    # Get top performer from best bucket
    best_bucket = ranked_buckets[0]['bucket']
    best_bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{best_bucket}')

    qr_video = select_qr_code_videos(best_bucket_path, "top")

    # Generate QR code
    qr_output_dir = os.path.join(reports_base_path, 'qr_codes')
    os.makedirs(qr_output_dir, exist_ok=True)
    qr_data = [{
        "filename": f"{args.competitor}_top.png",
        "url": qr_video['url']
    }]
    generate_qr_codes(qr_data, qr_output_dir)

    # =============================
    # STEP 8: Build Excel Data Structure
    # =============================
    tab_data = []

    # PAGE 1: EXECUTIVE OVERVIEW
    tab_data.append(['PAGE_1_EXECUTIVE_OVERVIEW', ''])
    tab_data.append(['', ''])

    tab_data.append(['COMPETITOR_HANDLE', f'@{args.competitor}'])
    tab_data.append(['ANALYSIS_PERIOD', 'Last 90 days'])
    tab_data.append(['VIDEOS_ANALYZED', str(total_videos)])

    # PAGE 2: CONTENT STRATEGY ANALYSIS
    tab_data.append(['', ''])
    tab_data.append(['PAGE_2_CONTENT_STRATEGY', ''])
    tab_data.append(['', ''])

    # Duration Distribution
    all_buckets = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
    for bucket in all_buckets:
        field_name = f'BUCKET_{bucket.replace("-", "_").upper()}_PCT'
        pct = bucket_percentages.get(bucket, 0)
        tab_data.append([field_name, str(pct)])

    tab_data.append(['', ''])
    tab_data.append(['PRIMARY_FOCUS_BUCKET', primary_focus])

    # Performance by Duration
    tab_data.append(['', ''])
    for i, bucket_data in enumerate(ranked_buckets, 1):
        tab_data.append([f'PERF_BUCKET_{i}_NAME', bucket_data['bucket']])
        tab_data.append([f'PERF_BUCKET_{i}_AVG_VIEWS', format_views(bucket_data['avg_views'])])
        tab_data.append([f'PERF_BUCKET_{i}_AVG_ENG', str(bucket_data['avg_engagement'])])
        tab_data.append([f'PERF_BUCKET_{i}_STARS', bucket_data['stars']])
        tab_data.append([f'PERF_BUCKET_{i}_IS_SWEET_SPOT', str(bucket_data['is_sweet_spot'])])
        tab_data.append(['', ''])

    tab_data.append(['SWEET_SPOT_BUCKET', sweet_spot])
    tab_data.append(['COVERAGE_PERCENTAGE', str(coverage_pct)])

    # Posting Frequency
    tab_data.append(['', ''])
    tab_data.append(['POSTING_FREQUENCY', str(posting_freq)])

    # PAGE 3: CREATIVE INTELLIGENCE
    tab_data.append(['', ''])
    tab_data.append(['PAGE_3_CREATIVE_INTELLIGENCE', ''])
    tab_data.append(['', ''])

    # Content Categories
    for i, (category, count) in enumerate(top_5_categories, 1):
        pct = round((count / total_classified) * 100)
        tab_data.append([f'CONTENT_CATEGORY_{i}', category.replace('_', ' ').title()])
        tab_data.append([f'CONTENT_CATEGORY_{i}_PCT', str(pct)])
        tab_data.append([f'CONTENT_CATEGORY_{i}_DESC', 'Description placeholder'])  # Would come from taxonomy

    tab_data.append(['', ''])

    # Engagement Drivers
    for i, (driver, count) in enumerate(top_4_drivers, 1):
        pct = round((count / total_classified) * 100)
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}', driver.replace('_', ' ').title()])
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}_PCT', str(pct)])
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}_DESC', 'Description placeholder'])

    # Hook Strategies
    tab_data.append(['', ''])
    for i, (hook, count) in enumerate(top_4_hooks, 1):
        pct = round((count / total_classified) * 100)
        tab_data.append([f'HOOK_STRATEGY_{i}', hook.replace('_', ' ').title()])
        tab_data.append([f'HOOK_STRATEGY_{i}_PCT', str(pct)])
        tab_data.append([f'HOOK_STRATEGY_{i}_DESC', 'Description placeholder'])

    tab_data.append(['', ''])

    # CTA Strategies (from closing_strategy field - no descriptions in taxonomy)
    for i, (cta, count) in enumerate(top_4_ctas, 1):
        pct = round((count / total_classified) * 100)
        tab_data.append([f'CTA_STRATEGY_{i}', cta.replace('_', ' ').title()])
        tab_data.append([f'CTA_STRATEGY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Pain Points
    for i, (pain, count) in enumerate(top_5_pain_points, 1):
        pct = round((count / total_classified) * 100)
        tab_data.append([f'PAIN_POINT_{i}', pain.replace('_', ' ').title()])
        tab_data.append([f'PAIN_POINT_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Keywords
    for i, (keyword, count) in enumerate(top_5_keywords, 1):
        tab_data.append([f'KEYWORD_{i}', keyword])

    tab_data.append(['', ''])

    # Content Tactics
    for i, (tactic, count) in enumerate(top_4_tactics, 1):
        pct = round((count / total_classified) * 100)
        tab_data.append([f'CONTENT_TACTIC_{i}', tactic.replace('_', ' ').title()])
        tab_data.append([f'CONTENT_TACTIC_{i}_PCT', str(pct)])

    # Hashtag Strategy
    tab_data.append(['', ''])
    for i, hashtag_data in enumerate(hashtag_analysis['top_10_hashtags'], 1):
        tab_data.append([f'HASHTAG_{i}', hashtag_data['tag']])
        tab_data.append([f'HASHTAG_{i}_PCT', str(hashtag_data['usage_pct'])])

    tab_data.append(['', ''])
    tab_data.append(['TOTAL_UNIQUE_HASHTAGS', str(hashtag_analysis['total_unique_hashtags'])])
    tab_data.append(['AVG_HASHTAGS_PER_VIDEO', str(int(hashtag_analysis['avg_hashtags_per_video']))])
    tab_data.append(['STRATEGY_TYPE', strategy_type])

    # Caption Strategy
    tab_data.append(['', ''])
    tab_data.append(['AVG_HASHTAG_COUNT', str(int(hashtag_analysis['avg_hashtags_per_video']))])

    if all_cta_types:
        top_cta, top_cta_count = all_cta_types.most_common(1)[0]
        top_cta_pct = round((top_cta_count / total_classified) * 100)
        tab_data.append(['TOP_CTA_TYPE', top_cta.replace('_', ' ').title()])
        tab_data.append(['TOP_CTA_TYPE_PCT', str(top_cta_pct)])

    # Content Sourcing
    tab_data.append(['', ''])
    tab_data.append(['ORIGINAL_CONTENT_PCT', str(original_content_pct)])
    tab_data.append(['REPOSTED_AFFILIATE_PCT', str(mention_analysis['repost_rate'])])
    tab_data.append(['', ''])

    # Top 5 affiliates
    for i, affiliate in enumerate(mention_analysis['top_10_mentions'][:5], 1):
        tab_data.append([f'AFFILIATE_{i}_HANDLE', affiliate['handle']])
        tab_data.append([f'AFFILIATE_{i}_PCT', str(affiliate['percentage'])])
        tab_data.append([f'AFFILIATE_{i}_COUNT', str(affiliate['mention_count'])])

    tab_data.append(['', ''])
    tab_data.append(['TOTAL_UNIQUE_MENTIONS', str(mention_analysis['total_unique_mentions'])])

    # Section 6: Creative Formulas
    tab_data.append(['', ''])
    # Extract bucket names and formulas from winning_formulas.json
    for i, bucket_name in enumerate(winner_data['top_3_buckets'], 1):
        bucket_path = os.path.join(base_path, 'buckets', f'bucket_{bucket_name}')
        winning_formulas_path = os.path.join(bucket_path, 'ml_analysis', 'llm', 'winning_formulas.json')

        tab_data.append([f'BUCKET_{i}_NAME', bucket_name])

        if os.path.exists(winning_formulas_path):
            with open(winning_formulas_path, 'r') as f:
                winning_formulas = json.load(f)
                creative_reports = winning_formulas.get('creative_reports', [])

                for j in range(min(3, len(creative_reports))):
                    formula_name = creative_reports[j].get('formula_name', '')
                    tab_data.append([f'BUCKET_{i}_FORMULA_{j+1}_NAME', formula_name])

        if i < 3:  # Add empty row between buckets (not after last one)
            tab_data.append(['', ''])

    # QR Code Metadata
    tab_data.append(['', ''])
    tab_data.append(['QR_CODE_FILE', f"{args.competitor}_top.png"])
    tab_data.append(['QR_CODE_URL', qr_video['url']])
    tab_data.append(['QR_CODE_VIEWS', format_views(qr_video['views'])])
    tab_data.append(['QR_CODE_ENGAGEMENT', str(qr_video['engagement'])])
    tab_data.append(['QR_CODE_DURATION', f"{qr_video['duration']}s"])
    tab_data.append(['QR_CODE_BUCKET', best_bucket])

    # =============================
    # STEP 9: Write Excel File
    # =============================
    excel_filename = f"{args.competitor}_analysis_data.xlsx"
    excel_path = os.path.join(reports_base_path, excel_filename)

    df = pd.DataFrame(tab_data, columns=['Field Name', 'Value'])
    df.to_excel(excel_path, sheet_name='Report_Data', index=False, engine='openpyxl')

    # =============================
    # STEP 10: Print Success Message
    # =============================
    print(f"\n✓ Extraction complete")
    print(f"  Excel: {excel_path}")
    print(f"  QR code: {os.path.join(qr_output_dir, qr_data[0]['filename'])}")
    print(f"  Total fields: {len(tab_data)}")


if __name__ == '__main__':
    main()
```

---

### Implementation Notes for Developer

**TODO items in skeleton above**:
1. ✅ All core functions implemented (engagement, bucket distribution, ranking, aggregation)
2. ✅ Hashtag and mention analysis functions complete
3. ✅ QR code generation implemented
4. ⚠️ Taxonomy descriptions for categories/hooks/drivers - requires Stage 2.6 taxonomy files or hardcoded mappings

**Testing checklist**:
- [ ] Script runs without errors
- [ ] Excel file created with single tab
- [ ] All ~140 fields populated (no empty values except intentional empty rows)
- [ ] QR code PNG generated (290x290px, ~5KB)
- [ ] QR code scans correctly to TikTok video URL
- [ ] Field values match source JSON files
- [ ] Percentages sum correctly
- [ ] Star ratings in correct order (rank 1 = 5 stars, sweet spot = True)
- [ ] Hashtag analysis accurate (top 10 with usage %)
- [ ] Mention analysis accurate (affiliate detection)

**Error handling**:
Script should exit with clear error if:
- `winner_analysis.json` not found
- `selected_videos.json` not found for any winning bucket
- JSON files malformed
- Analysis directory discovery fails
- Cannot write Excel file (permissions)
- Cannot create qr_codes directory
- QR code generation fails

**Dependencies**:
```bash
pip install pandas openpyxl qrcode[pil]
```

---

**END OF SECTION 3.3**

This section is complete and self-contained for implementation of `extract_competitor_data.py`.

---

## Section 3.4: `extract_multi_competitor_data.py` - COMPLETE IMPLEMENTATION GUIDE

### Overview

**Purpose**: Extract multi-competitor market intelligence data for client executive report (Report 4)

**Report Type**: Report 4 from Stage8MVP_Reports.md Section 4

**Deliverable**: 1 Excel file + (N competitors × 2 QR codes per winning bucket × 3 buckets)

**CLI Usage**:
```bash
python extract_multi_competitor_data.py --client acme --competitors drinkpoppi,nike,vitalproteins --mode top --strategy contrastive
```

**Output Files**:
```
/data/clients/acme/market_intelligence/multi_competitor/
├── market_intelligence_report.xlsx (single tab with all pages)
└── qr_codes/
    ├── drinkpoppi_18-33s_rank1.png
    ├── drinkpoppi_18-33s_rank2.png
    ├── drinkpoppi_13-18s_rank1.png
    ├── drinkpoppi_13-18s_rank2.png
    ├── drinkpoppi_33-60s_rank1.png
    ├── drinkpoppi_33-60s_rank2.png
    ├── nike_18-33s_rank1.png
    ├── nike_18-33s_rank2.png
    ├── [... 6 QR codes per competitor × 3 competitors = 18 total]
    └── vitalproteins_13-18s_rank2.png
```

**Console Output Pattern**:
```bash
$ python extract_multi_competitor_data.py --client acme --competitors drinkpoppi,nike,vitalproteins

Running multi-competitor extraction
Analyzing 3 competitors: @drinkpoppi, @nike, @vitalproteins
Loading performance data for all competitors...
Building bucket distribution matrix (8 buckets × 3 competitors)...
Building performance matrix (5 unique winning buckets × 3 competitors)...
Aggregating per-bucket content intelligence (9 bucket-competitor combinations)...
Extracting hashtag and mention analysis for 3 competitors...
Generating 18 QR codes (2 per bucket × 3 buckets × 3 competitors)...

✓ Extraction complete
  Excel: /data/clients/acme/market_intelligence/multi_competitor/market_intelligence_report.xlsx
  QR codes: /data/clients/acme/market_intelligence/multi_competitor/qr_codes/ (18 files)
  Total fields: ~350-400 (varies by competitor count and QR codes)
```

---

### Complete Field List

**Excel Structure**: Single tab with two-column format (Field Name | Value)

**Total Fields**: ~150-300 fields (varies by competitor count: 2-5)

**Note**: Field examples below show 3 competitors. Actual implementation scales to 2-5 competitors.

```python
# Field structure - two-column format: Field Name | Value
# Example with 3 competitors: @drinkpoppi, @nike, @vitalproteins

fields = [
    # =============================
    # PAGE 1: MARKET OVERVIEW
    # =============================
    ('PAGE_1_MARKET_OVERVIEW', ''),  # Section divider
    ('', ''),

    # --- Header Section ---
    ('COMPETITOR_COUNT', '3'),
    ('COMPETITOR_1_HANDLE', '@drinkpoppi'),
    ('COMPETITOR_2_HANDLE', '@nike'),
    ('COMPETITOR_3_HANDLE', '@vitalproteins'),
    ('ANALYSIS_PERIOD', 'Last 90 days'),  # Static
    ('', ''),

    # --- Performance Rankings (sorted by composite score) ---
    ('RANK_1_HANDLE', '@nike'),
    ('RANK_1_AVG_VIEWS', '620K'),
    ('RANK_1_AVG_ENGAGEMENT', '1.5'),
    ('RANK_1_POSTING_FREQ', '16'),
    ('RANK_1_VIDEOS_ANALYZED', '145'),
    ('', ''),

    ('RANK_2_HANDLE', '@vitalproteins'),
    ('RANK_2_AVG_VIEWS', '580K'),
    ('RANK_2_AVG_ENGAGEMENT', '1.4'),
    ('RANK_2_POSTING_FREQ', '14'),
    ('RANK_2_VIDEOS_ANALYZED', '127'),
    ('', ''),

    ('RANK_3_HANDLE', '@drinkpoppi'),
    ('RANK_3_AVG_VIEWS', '520K'),
    ('RANK_3_AVG_ENGAGEMENT', '1.3'),
    ('RANK_3_POSTING_FREQ', '11'),
    ('RANK_3_VIDEOS_ANALYZED', '98'),
    ('', ''),

    ('MARKET_LEADER', '@nike'),
    ('MARKET_LEADER_REASON', '620K avg views, 1.5% engagement, highest posting frequency'),

    # --- Analysis Scope (per competitor) ---
    ('', ''),
    ('COMP_1_VIDEOS_ANALYZED', '145'),
    ('COMP_2_VIDEOS_ANALYZED', '127'),
    ('COMP_3_VIDEOS_ANALYZED', '98'),

    # =============================
    # PAGE 2: CONTENT STRATEGY COMPARISON
    # =============================
    ('', ''),
    ('PAGE_2_CONTENT_STRATEGY', ''),
    ('', ''),

    # --- Bucket Distribution Matrix (8 buckets × 3 competitors) ---
    # Format: BUCKET_{bucket}_COMP_{n}_PCT
    ('BUCKET_0_3S_COMP_1_PCT', '2'),
    ('BUCKET_0_3S_COMP_2_PCT', '3'),
    ('BUCKET_0_3S_COMP_3_PCT', '5'),
    ('BUCKET_0_3S_MARKET_PATTERN', 'Low volume'),
    ('', ''),

    ('BUCKET_3_9S_COMP_1_PCT', '5'),
    ('BUCKET_3_9S_COMP_2_PCT', '8'),
    ('BUCKET_3_9S_COMP_3_PCT', '10'),
    ('BUCKET_3_9S_MARKET_PATTERN', 'Low volume'),
    ('', ''),

    ('BUCKET_9_13S_COMP_1_PCT', '8'),
    ('BUCKET_9_13S_COMP_2_PCT', '12'),
    ('BUCKET_9_13S_COMP_3_PCT', '14'),
    ('BUCKET_9_13S_MARKET_PATTERN', 'Growing volume'),
    ('', ''),

    ('BUCKET_13_18S_COMP_1_PCT', '15'),
    ('BUCKET_13_18S_COMP_2_PCT', '18'),
    ('BUCKET_13_18S_COMP_3_PCT', '22'),
    ('BUCKET_13_18S_MARKET_PATTERN', 'Moderate volume'),
    ('', ''),

    ('BUCKET_18_33S_COMP_1_PCT', '28'),  # High volume (>20%)
    ('BUCKET_18_33S_COMP_1_HIGH_VOLUME', 'True'),
    ('BUCKET_18_33S_COMP_2_PCT', '32'),
    ('BUCKET_18_33S_COMP_2_HIGH_VOLUME', 'True'),
    ('BUCKET_18_33S_COMP_3_PCT', '26'),
    ('BUCKET_18_33S_COMP_3_HIGH_VOLUME', 'True'),
    ('BUCKET_18_33S_MARKET_PATTERN', 'HIGH VOLUME'),
    ('', ''),

    ('BUCKET_33_60S_COMP_1_PCT', '30'),
    ('BUCKET_33_60S_COMP_1_HIGH_VOLUME', 'True'),
    ('BUCKET_33_60S_COMP_2_PCT', '22'),
    ('BUCKET_33_60S_COMP_2_HIGH_VOLUME', 'True'),
    ('BUCKET_33_60S_COMP_3_PCT', '18'),
    ('BUCKET_33_60S_COMP_3_HIGH_VOLUME', 'False'),
    ('BUCKET_33_60S_MARKET_PATTERN', 'High volume'),
    ('', ''),

    ('BUCKET_60_90S_COMP_1_PCT', '9'),
    ('BUCKET_60_90S_COMP_1_HIGH_VOLUME', 'False'),
    ('BUCKET_60_90S_COMP_2_PCT', '4'),
    ('BUCKET_60_90S_COMP_2_HIGH_VOLUME', 'False'),
    ('BUCKET_60_90S_COMP_3_PCT', '4'),
    ('BUCKET_60_90S_COMP_3_HIGH_VOLUME', 'False'),
    ('BUCKET_60_90S_MARKET_PATTERN', 'Low volume'),
    ('', ''),

    ('BUCKET_90_120S_COMP_1_PCT', '3'),
    ('BUCKET_90_120S_COMP_1_HIGH_VOLUME', 'False'),
    ('BUCKET_90_120S_COMP_2_PCT', '1'),
    ('BUCKET_90_120S_COMP_2_HIGH_VOLUME', 'False'),
    ('BUCKET_90_120S_COMP_3_PCT', '1'),
    ('BUCKET_90_120S_COMP_3_HIGH_VOLUME', 'False'),
    ('BUCKET_90_120S_MARKET_PATTERN', 'Very low volume'),

    # --- Performance Matrix (unique winning buckets × 3 competitors) ---
    # Union of all winning buckets: ["9-13s", "13-18s", "18-33s", "33-60s", "60-90s"]
    ('', ''),
    ('UNIQUE_WINNING_BUCKETS_COUNT', '5'),
    ('UNIQUE_WINNING_BUCKET_1', '9-13s'),
    ('UNIQUE_WINNING_BUCKET_2', '13-18s'),
    ('UNIQUE_WINNING_BUCKET_3', '18-33s'),
    ('UNIQUE_WINNING_BUCKET_4', '33-60s'),
    ('UNIQUE_WINNING_BUCKET_5', '60-90s'),
    ('', ''),

    # Format: PERF_{bucket}_{metric}_COMP_{n}
    ('PERF_9_13S_VIEWS_COMP_1', '420K'),
    ('PERF_9_13S_ENGAGEMENT_COMP_1', '1.2'),
    ('PERF_9_13S_WINNING_COMP_1', 'True'),  # In this competitor's top 3
    ('PERF_9_13S_VIEWS_COMP_2', '—'),  # Not a winning bucket
    ('PERF_9_13S_ENGAGEMENT_COMP_2', '—'),
    ('PERF_9_13S_WINNING_COMP_2', 'False'),
    ('PERF_9_13S_VIEWS_COMP_3', '—'),
    ('PERF_9_13S_ENGAGEMENT_COMP_3', '—'),
    ('PERF_9_13S_WINNING_COMP_3', 'False'),
    ('PERF_9_13S_BEST_PERFORMER', '@nike'),
    ('', ''),

    ('PERF_13_18S_VIEWS_COMP_1', '580K'),
    ('PERF_13_18S_ENGAGEMENT_COMP_1', '1.3'),
    ('PERF_13_18S_WINNING_COMP_1', 'True'),
    ('PERF_13_18S_VIEWS_COMP_2', '580K'),
    ('PERF_13_18S_ENGAGEMENT_COMP_2', '1.4'),
    ('PERF_13_18S_WINNING_COMP_2', 'True'),
    ('PERF_13_18S_VIEWS_COMP_3', '490K'),
    ('PERF_13_18S_ENGAGEMENT_COMP_3', '1.2'),
    ('PERF_13_18S_WINNING_COMP_3', 'True'),
    ('PERF_13_18S_BEST_PERFORMER', '@vitalproteins (engagement wins tie)'),
    # ... similar for other unique buckets

    # --- Posting Frequency ---
    ('', ''),
    ('POSTING_FREQ_COMP_1', '16'),
    ('POSTING_FREQ_COMP_2', '14'),
    ('POSTING_FREQ_COMP_3', '11'),
    ('MARKET_AVG_POSTING_FREQ', '13.7'),

    # =============================
    # PAGE 3: CREATIVE INTELLIGENCE (PER-BUCKET)
    # =============================
    ('', ''),
    ('PAGE_3_CREATIVE_INTELLIGENCE', ''),
    ('', ''),

    # Note: Per-bucket aggregations for 3 winning buckets × 3 competitors = 9 combinations
    # Format: {FIELD}_{COMPETITOR}_{BUCKET}_{INDEX}

    # --- Content Categories (Top 2 per bucket per competitor) ---
    # Competitor 1, Bucket 1 (18-33s)
    ('CONTENT_CAT_COMP_1_BUCKET_18_33S_1', 'Recipe Tutorial'),
    ('CONTENT_CAT_COMP_1_BUCKET_18_33S_2', 'Wellness Practice'),
    # Competitor 1, Bucket 2 (13-18s)
    ('CONTENT_CAT_COMP_1_BUCKET_13_18S_1', 'Supplement Review'),
    ('CONTENT_CAT_COMP_1_BUCKET_13_18S_2', 'Expert Interview'),
    # Competitor 1, Bucket 3 (33-60s)
    ('CONTENT_CAT_COMP_1_BUCKET_33_60S_1', 'Personal Testimony'),
    ('CONTENT_CAT_COMP_1_BUCKET_33_60S_2', 'Recipe Tutorial'),
    ('', ''),

    # Competitor 2, Buckets 1-3
    ('CONTENT_CAT_COMP_2_BUCKET_18_33S_1', 'Recipe Tutorial'),
    ('CONTENT_CAT_COMP_2_BUCKET_18_33S_2', 'Product Demo'),
    # ... similar for other buckets

    # Competitor 3, Buckets 1-3
    # ... similar structure

    # --- Engagement Drivers (Top 2 per bucket per competitor) ---
    ('ENGAGEMENT_DRIVER_COMP_1_BUCKET_18_33S_1', 'Before/After Reveal'),
    ('ENGAGEMENT_DRIVER_COMP_1_BUCKET_18_33S_2', 'Personal Testimony'),
    # ... similar for all bucket-competitor combinations

    # --- Hook Strategies (Top 2 per bucket per competitor) ---
    ('HOOK_STRATEGY_COMP_1_BUCKET_18_33S_1', 'Question Hook'),
    ('HOOK_STRATEGY_COMP_1_BUCKET_18_33S_2', 'Problem-Solution'),
    # ... similar for all combinations

    # --- CTA Strategies (Top 2 per bucket per competitor) ---
    # Note: From closing_strategy field (video ending CTA, not caption)
    ('CTA_STRATEGY_COMP_1_BUCKET_18_33S_1', 'Declarative Statement'),
    ('CTA_STRATEGY_COMP_1_BUCKET_18_33S_2', 'Question'),
    # ... similar for all combinations

    # --- Caption CTA Strategies (Top 2 per bucket per competitor) ---
    # Note: From caption_cta_type field (written caption call-to-action)
    ('CAPTION_CTA_STRATEGY_COMP_1_BUCKET_18_33S_1', 'Link in Bio'),
    ('CAPTION_CTA_STRATEGY_COMP_1_BUCKET_18_33S_2', 'Save This Post'),
    # ... similar for all combinations

    # --- Pain Points (Top 3 per bucket per competitor) ---
    ('PAIN_POINT_COMP_1_BUCKET_18_33S_1', 'Bloating'),
    ('PAIN_POINT_COMP_1_BUCKET_18_33S_2', 'Low Energy'),
    ('PAIN_POINT_COMP_1_BUCKET_18_33S_3', 'Weight Management'),
    # ... similar for all combinations

    # --- Keywords (Top 3 per bucket per competitor) ---
    ('KEYWORD_COMP_1_BUCKET_18_33S_1', 'gut health'),
    ('KEYWORD_COMP_1_BUCKET_18_33S_2', 'protein'),
    ('KEYWORD_COMP_1_BUCKET_18_33S_3', 'fiber'),
    # ... similar for all combinations

    # --- Content Tactics (Top 2 per bucket per competitor) ---
    ('CONTENT_TACTIC_COMP_1_BUCKET_18_33S_1', 'Direct-to-Camera'),
    ('CONTENT_TACTIC_COMP_1_BUCKET_18_33S_2', 'Voiceover'),
    # ... similar for all combinations

    # --- Supplementary Insights (Top 5 per bucket per competitor) ---
    # Only included if competitor has this bucket in their top_3_buckets
    ('', ''),
    # Bucket 1 (18-33s) - Competitor 1
    ('SUPP_INSIGHT_COMP_1_BUCKET_18_33S_1', 'middle_3_eye_contact_rate: 0.57 in top vs 0.43 in bottom (gap: 0.14)'),
    ('SUPP_INSIGHT_COMP_1_BUCKET_18_33S_2', 'middle_1_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)'),
    ('SUPP_INSIGHT_COMP_1_BUCKET_18_33S_3', 'middle_3_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)'),
    ('SUPP_INSIGHT_COMP_1_BUCKET_18_33S_4', 'middle_3_energy_level: 0.10 in top vs 0.06 in bottom (gap: 0.04)'),
    ('SUPP_INSIGHT_COMP_1_BUCKET_18_33S_5', 'hook_eye_contact_rate: 0.51 in top vs 0.63 in bottom (gap: 0.11)'),
    # Bucket 1 (18-33s) - Competitor 2 (if applicable)
    ('SUPP_INSIGHT_COMP_2_BUCKET_18_33S_1', 'middle_2_word_count: 25.3 in top vs 18.2 in bottom (gap: 7.1)'),
    ('SUPP_INSIGHT_COMP_2_BUCKET_18_33S_2', '...'),
    ('SUPP_INSIGHT_COMP_2_BUCKET_18_33S_3', '...'),
    ('SUPP_INSIGHT_COMP_2_BUCKET_18_33S_4', '...'),
    ('SUPP_INSIGHT_COMP_2_BUCKET_18_33S_5', '...'),
    # Bucket 1 (18-33s) - Competitor 3 (if applicable)
    ('SUPP_INSIGHT_COMP_3_BUCKET_18_33S_1', '...'),
    ('SUPP_INSIGHT_COMP_3_BUCKET_18_33S_2', '...'),
    ('SUPP_INSIGHT_COMP_3_BUCKET_18_33S_3', '...'),
    ('SUPP_INSIGHT_COMP_3_BUCKET_18_33S_4', '...'),
    ('SUPP_INSIGHT_COMP_3_BUCKET_18_33S_5', '...'),
    ('', ''),
    # Bucket 2 (13-18s) - Competitors (if bucket is common)
    ('SUPP_INSIGHT_COMP_1_BUCKET_13_18S_1', '...'),
    # ... similar for all applicable competitors
    ('', ''),
    # Bucket 3 (33-60s) - Competitors (if bucket is common)
    ('SUPP_INSIGHT_COMP_1_BUCKET_33_60S_1', '...'),
    # ... similar for all applicable competitors

    # --- Hashtag Strategy (aggregate across all buckets per competitor) ---
    ('', ''),
    ('HASHTAG_TOTAL_UNIQUE_COMP_1', '42'),
    ('HASHTAG_AVG_PER_VIDEO_COMP_1', '11'),
    ('HASHTAG_TOP_5_CONCENTRATION_COMP_1', '65'),
    ('HASHTAG_STRATEGY_TYPE_COMP_1', 'Diversified'),
    # Top 5 hashtags for Competitor 1
    ('HASHTAG_COMP_1_1', '#wellness'),
    ('HASHTAG_COMP_1_1_PCT', '78'),
    ('HASHTAG_COMP_1_2', '#healthylifestyle'),
    ('HASHTAG_COMP_1_2_PCT', '68'),
    ('HASHTAG_COMP_1_3', '#nutrition'),
    ('HASHTAG_COMP_1_3_PCT', '62'),
    ('HASHTAG_COMP_1_4', '#guthealth'),
    ('HASHTAG_COMP_1_4_PCT', '55'),
    ('HASHTAG_COMP_1_5', '#holistichealth'),
    ('HASHTAG_COMP_1_5_PCT', '48'),
    ('', ''),

    # Similar for Competitors 2 and 3
    ('HASHTAG_TOTAL_UNIQUE_COMP_2', '28'),
    # ... full hashtag data for Competitor 2

    ('HASHTAG_TOTAL_UNIQUE_COMP_3', '35'),
    # ... full hashtag data for Competitor 3

    # --- Caption Strategy (aggregate across all buckets per competitor) ---
    ('', ''),
    ('CAPTION_AVG_HASHTAG_COUNT_COMP_1', '11'),
    ('CAPTION_TOP_CTA_COMP_1', 'Link in bio'),
    ('CAPTION_TOP_CTA_COMP_1_PCT', '68'),

    ('CAPTION_AVG_HASHTAG_COUNT_COMP_2', '9'),
    ('CAPTION_TOP_CTA_COMP_2', 'Link in bio'),
    ('CAPTION_TOP_CTA_COMP_2_PCT', '72'),

    ('CAPTION_AVG_HASHTAG_COUNT_COMP_3', '10'),
    ('CAPTION_TOP_CTA_COMP_3', 'Follow'),
    ('CAPTION_TOP_CTA_COMP_3_PCT', '58'),

    # --- Content Sourcing Strategy (per competitor) ---
    ('', ''),
    ('SOURCING_UGC_PCT_COMP_1', '28'),
    ('SOURCING_OWN_PCT_COMP_1', '72'),
    ('SOURCING_UNIQUE_AFFILIATES_COMP_1', '22'),
    # Top 4 affiliates for Competitor 1 (with full brand names extracted from captions)
    ('AFFILIATE_COMP_1_1_HANDLE', '@holistichealth_coach (Holistic Health Coach)'),
    ('AFFILIATE_COMP_1_1_PCT', '12'),
    ('AFFILIATE_COMP_1_1_COUNT', '36'),
    ('AFFILIATE_COMP_1_2_HANDLE', '@wellness_collective (Wellness Collective)'),
    ('AFFILIATE_COMP_1_2_PCT', '8'),
    ('AFFILIATE_COMP_1_2_COUNT', '24'),
    ('AFFILIATE_COMP_1_3_HANDLE', '@naturalremedies (Natural Remedies)'),
    ('AFFILIATE_COMP_1_3_PCT', '5'),
    ('AFFILIATE_COMP_1_3_COUNT', '15'),
    ('AFFILIATE_COMP_1_4_HANDLE', '@ayurveda_lifestyle (Ayurveda Lifestyle)'),
    ('AFFILIATE_COMP_1_4_PCT', '3'),
    ('AFFILIATE_COMP_1_4_COUNT', '9'),
    ('', ''),

    # Similar for Competitors 2 and 3
    ('SOURCING_UGC_PCT_COMP_2', '42'),
    # ... full sourcing data for Competitor 2

    ('SOURCING_UGC_PCT_COMP_3', '15'),
    # ... full sourcing data for Competitor 3

    # =============================
    # PAGE 4: VISUAL EXAMPLES (QR CODES)
    # =============================
    ('', ''),
    ('PAGE_4_VISUAL_EXAMPLES', ''),
    ('', ''),

    # QR Code metadata for each competitor
    ('QR_COMP_1_FILE', 'drinkpoppi_top.png'),
    ('QR_COMP_1_URL', 'https://www.tiktok.com/@drinkpoppi/video/7540717847325003039'),
    ('QR_COMP_1_VIEWS', '820K'),
    ('QR_COMP_1_ENGAGEMENT', '1.5'),
    ('QR_COMP_1_DURATION', '45s'),
    ('QR_COMP_1_BUCKET', '33-60s'),
    ('QR_COMP_1_HASHTAGS', '#wellness #guthealth #transformation #healthylifestyle'),
    ('', ''),

    ('QR_COMP_2_FILE', 'nike_top.png'),
    ('QR_COMP_2_URL', 'https://www.tiktok.com/@nike/video/7540717847325003040'),
    ('QR_COMP_2_VIEWS', '720K'),
    ('QR_COMP_2_ENGAGEMENT', '1.4'),
    ('QR_COMP_2_DURATION', '22s'),
    ('QR_COMP_2_BUCKET', '18-33s'),
    ('QR_COMP_2_HASHTAGS', '#nutrition #guthealth #recipe #protein'),
    ('', ''),

    ('QR_COMP_3_FILE', 'vitalproteins_top.png'),
    ('QR_COMP_3_URL', 'https://www.tiktok.com/@vitalproteins/video/7540717847325003041'),
    ('QR_COMP_3_VIEWS', '650K'),
    ('QR_COMP_3_ENGAGEMENT', '1.3'),
    ('QR_COMP_3_DURATION', '16s'),
    ('QR_COMP_3_BUCKET', '13-18s'),
    ('QR_COMP_3_HASHTAGS', '#fitness #supplements #protein #healthytips'),
]
```

**Notes**:
- Total fields: ~150-300 depending on competitor count (2-5)
- 3 competitors example: ~287 fields (+ variable supplementary insights fields)
- Per-bucket aggregations: (3 buckets × N competitors) × (2 categories + 2 drivers + 2 hooks + 2 CTAs + 3 pain points + 3 keywords + 2 tactics) = significant field expansion
- Supplementary Insights: Variable field count based on common buckets
- Field naming: `{TYPE}_{COMP|BUCKET}_{INDEX}`
- Empty rows for visual separation

---

### Implementation Notes for Supplementary Insights

**Challenge**: Variable structure based on which competitors have which winning buckets

**Logic**:
1. **Bucket Discovery**:
   - Load each competitor's `winner_analysis.json`
   - Extract `top_3_buckets` for each competitor
   - Find common buckets (buckets appearing in 2+ competitors' lists)
   - Sort by frequency (most common first)

2. **Conditional Field Generation**:
   - Only create fields for common buckets
   - Within each bucket, only include competitors who have that bucket in their `top_3_buckets`
   - Field naming: `SUPP_INSIGHT_COMP_{comp_idx}_BUCKET_{bucket_key}_{insight_num}`
   - Example: `SUPP_INSIGHT_COMP_1_BUCKET_18_33S_3`

3. **Field Count Calculation**:
   - Variable based on common buckets and applicable competitors
   - Formula: `sum(len(bucket_competitors[bucket]) * 5 for bucket in common_buckets)`
   - Example: 2 common buckets with [3, 2] competitors = (3×5) + (2×5) = 25 insight fields

4. **Edge Cases**:
   - If no common buckets exist across competitors → Section 5 not rendered (0 fields)
   - If `winning_formulas.json` missing → Skip that competitor for that bucket
   - If insights array has <5 items → Only populate available insights
   - Bucket name formatting: "18-33s" → "18_33S" for field name

5. **Maximum Fields**: 3 buckets × 5 competitors × 5 insights = 75 fields (extreme case)
6. **Typical Fields**: 2 buckets × 3 competitors × 5 insights = 30 fields

---

### Required Functions

This section defines all functions needed for `extract_multi_competitor_data.py`. Functions build on those from Reports 1-3.

---

#### Function 1: `calculate_engagement_metrics()`

**Purpose**: Calculate real engagement rate from TikTok video metadata

**Implementation**: Same as Reports 1-3 - see Section 3.2 Function 1

---

#### Function 2: `calculate_competitor_rankings()`

**Purpose**: Rank all competitors by performance (composite score)

**Used by**: Report 4 (Page 1 Performance Rankings)

**Input**: Client ID and list of competitor handles

**Output**: List of dicts with competitor rankings

**Implementation**:
```python
def calculate_competitor_rankings(client_id, competitors):
    """
    Rank competitors by performance (views + engagement composite score).

    Args:
        client_id: Client identifier
        competitors: List of competitor handles (without @)

    Returns:
        List of dicts sorted by rank (best first)

    Example:
        [
            {
                "rank": 1,
                "handle": "@nike",
                "avg_views": 620000,
                "avg_engagement": 1.5,
                "posting_freq": 16.0,
                "videos_analyzed": 145,
                "composite_score": 101.5,
                "is_market_leader": True
            },
            ...
        ]
    """
    import os
    import json

    competitor_data = []

    for competitor in competitors:
        # Discover analysis directory
        base_path = f"/data/clients/{client_id}/competitors/{competitor}"
        analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]
        if not analysis_dirs:
            continue

        analysis_dir = analysis_dirs[0]
        competitor_path = f"{base_path}/{analysis_dir}"

        # Load winner analysis
        with open(f"{competitor_path}/winner_analysis.json") as f:
            winner_data = json.load(f)

        winning_buckets = winner_data["top_3_buckets"]

        # Calculate metrics
        total_views = 0
        total_engagement = 0
        total_videos = 0

        for bucket in winning_buckets:
            bucket_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"
            with open(bucket_path) as f:
                data = json.load(f)

            top_count = data["top_count"]
            top_videos = data["videos"][:top_count]

            for video in top_videos:
                total_views += video["playCount"]
                engagement = calculate_engagement_metrics(video)
                total_engagement += engagement
                total_videos += 1

        avg_views = int(total_views / total_videos) if total_videos > 0 else 0
        avg_engagement = round(total_engagement / total_videos, 1) if total_videos > 0 else 0.0

        # Posting frequency
        posting_freq = calculate_posting_frequency(client_id, competitor)

        competitor_data.append({
            "handle": f"@{competitor}",
            "avg_views": avg_views,
            "avg_engagement": avg_engagement,
            "posting_freq": posting_freq,
            "videos_analyzed": total_videos
        })

    # Calculate composite scores and rank
    if competitor_data:
        max_views = max(c["avg_views"] for c in competitor_data)

        for comp in competitor_data:
            normalized_views = (comp["avg_views"] / max_views) * 100
            composite_score = normalized_views + comp["avg_engagement"]
            comp["composite_score"] = composite_score

        # Sort by composite score DESC
        competitor_data.sort(key=lambda c: c["composite_score"], reverse=True)

        # Assign ranks
        for idx, comp in enumerate(competitor_data, start=1):
            comp["rank"] = idx
            comp["is_market_leader"] = (idx == 1)

    return competitor_data
```

---

#### Function 3: `build_bucket_distribution_matrix()`

**Purpose**: Build matrix of bucket distribution across all competitors

**Used by**: Report 4 (Page 2 Bucket Distribution Matrix)

**Input**: Client ID and list of competitor handles

**Output**: Dict with matrix data

**Implementation**:
```python
def build_bucket_distribution_matrix(client_id, competitors):
    """
    Build bucket distribution matrix: 8 buckets × N competitors.

    Args:
        client_id: Client identifier
        competitors: List of competitor handles

    Returns:
        dict: {
            "buckets": ["0-3s", "3-9s", ...],
            "matrix": {
                "0-3s": {
                    "competitors": [2, 3, 5],  # Percentages per competitor
                    "high_volume_markers": [False, False, False],
                    "market_pattern": "Low volume"
                },
                ...
            }
        }
    """
    import os
    import json

    all_buckets = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
    matrix = {}

    for bucket in all_buckets:
        bucket_data = {
            "competitors": [],
            "high_volume_markers": []
        }

        for competitor in competitors:
            # Get bucket percentage for this competitor
            base_path = f"/data/clients/{client_id}/competitors/{competitor}"
            analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]
            if not analysis_dirs:
                bucket_data["competitors"].append(0)
                bucket_data["high_volume_markers"].append(False)
                continue

            analysis_dir = analysis_dirs[0]
            winner_analysis_path = f"{base_path}/{analysis_dir}/winner_analysis.json"

            bucket_pct = calculate_bucket_distribution(winner_analysis_path).get(bucket, 0)
            bucket_data["competitors"].append(bucket_pct)

            # High volume marker (>20%)
            is_high_volume = (bucket_pct > 20)
            bucket_data["high_volume_markers"].append(is_high_volume)

        # Calculate market pattern
        avg_pct = sum(bucket_data["competitors"]) / len(bucket_data["competitors"]) if bucket_data["competitors"] else 0

        if avg_pct >= 25:
            market_pattern = "HIGH VOLUME"
        elif avg_pct >= 20:
            market_pattern = "High volume"
        elif avg_pct >= 15:
            market_pattern = "Moderate volume"
        elif avg_pct >= 10:
            market_pattern = "Growing volume"
        else:
            market_pattern = "Low volume"

        bucket_data["market_pattern"] = market_pattern
        matrix[bucket] = bucket_data

    return {
        "buckets": all_buckets,
        "matrix": matrix
    }
```

---

#### Function 4: `build_performance_matrix()`

**Purpose**: Build performance matrix for unique winning buckets across competitors

**Used by**: Report 4 (Page 2 Performance Matrix)

**Input**: Client ID and list of competitor handles

**Output**: Dict with performance matrix

**Implementation**:
```python
def build_performance_matrix(client_id, competitors):
    """
    Build performance matrix: unique winning buckets × N competitors.

    Only shows data for buckets that are in each competitor's top 3.

    Args:
        client_id: Client identifier
        competitors: List of competitor handles

    Returns:
        dict: {
            "unique_buckets": ["9-13s", "13-18s", ...],
            "matrix": {
                "9-13s": {
                    "competitors": [
                        {"handle": "@nike", "views": 420000, "engagement": 1.2, "is_winning": True},
                        {"handle": "@vita", "views": None, "engagement": None, "is_winning": False},
                        ...
                    ],
                    "best_performer": "@nike"
                },
                ...
            }
        }
    """
    import os
    import json

    # Step 1: Get union of all winning buckets
    all_winning_buckets = set()
    competitor_winning_buckets = {}

    for competitor in competitors:
        base_path = f"/data/clients/{client_id}/competitors/{competitor}"
        analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]
        if not analysis_dirs:
            continue

        analysis_dir = analysis_dirs[0]
        with open(f"{base_path}/{analysis_dir}/winner_analysis.json") as f:
            winner_data = json.load(f)

        winning_buckets = winner_data["top_3_buckets"]
        all_winning_buckets.update(winning_buckets)
        competitor_winning_buckets[competitor] = winning_buckets

    unique_buckets = sorted(list(all_winning_buckets), key=lambda b: ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"].index(b))

    # Step 2: Build matrix
    matrix = {}

    for bucket in unique_buckets:
        bucket_data = {"competitors": []}

        competitor_scores = []

        for competitor in competitors:
            is_winning = bucket in competitor_winning_buckets.get(competitor, [])

            if not is_winning:
                bucket_data["competitors"].append({
                    "handle": f"@{competitor}",
                    "views": None,
                    "engagement": None,
                    "is_winning": False
                })
                continue

            # Calculate metrics for this bucket
            base_path = f"/data/clients/{client_id}/competitors/{competitor}"
            analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]
            analysis_dir = analysis_dirs[0]
            competitor_path = f"{base_path}/{analysis_dir}"

            bucket_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"
            with open(bucket_path) as f:
                data = json.load(f)

            top_count = data["top_count"]
            top_videos = data["videos"][:top_count]

            avg_views = sum(v["playCount"] for v in top_videos) / len(top_videos)
            avg_engagement = sum(calculate_engagement_metrics(v) for v in top_videos) / len(top_videos)

            bucket_data["competitors"].append({
                "handle": f"@{competitor}",
                "views": int(avg_views),
                "engagement": round(avg_engagement, 1),
                "is_winning": True
            })

            # Track for best performer calculation
            competitor_scores.append({
                "handle": f"@{competitor}",
                "views": int(avg_views),
                "engagement": round(avg_engagement, 1)
            })

        # Determine best performer for this bucket
        if competitor_scores:
            max_views = max(c["views"] for c in competitor_scores)
            for comp in competitor_scores:
                normalized_views = (comp["views"] / max_views) * 100
                comp["composite_score"] = normalized_views + comp["engagement"]

            competitor_scores.sort(key=lambda c: c["composite_score"], reverse=True)
            best_performer = competitor_scores[0]["handle"]

            # Check for ties
            if len(competitor_scores) > 1 and competitor_scores[0]["views"] == competitor_scores[1]["views"]:
                best_performer += " (engagement wins tie)"

            bucket_data["best_performer"] = best_performer
        else:
            bucket_data["best_performer"] = "—"

        matrix[bucket] = bucket_data

    return {
        "unique_buckets": unique_buckets,
        "matrix": matrix
    }
```

---

#### Function 5: `aggregate_per_bucket_content()`

**Purpose**: Aggregate content intelligence per bucket for all competitors

**Used by**: Report 4 (Page 3 per-bucket content intelligence)

**Input**: Client ID and list of competitor handles

**Output**: Nested dict with per-bucket aggregations

**Implementation**:
```python
def aggregate_per_bucket_content(client_id, competitors):
    """
    Aggregate content intelligence per bucket per competitor.

    Args:
        client_id: Client identifier
        competitors: List of competitor handles

    Returns:
        dict: {
            "drinkpoppi": {
                "18-33s": {
                    "top_2_categories": ["recipe_tutorial", "wellness_practice"],
                    "top_2_drivers": ["before_after", "testimony"],
                    "top_2_hooks": ["question", "problem_solution"],
                    "top_2_ctas": ["link_bio", "save"],
                    "top_3_pain_points": ["bloating", "energy", "weight"],
                    "top_3_keywords": ["guthealth", "protein", "fiber"],
                    "top_2_tactics": ["direct_camera", "voiceover"]
                },
                "13-18s": {...},
                "33-60s": {...}
            },
            "nike": {...},
            "vitalproteins": {...}
        }
    """
    import os
    import json
    from collections import Counter

    results = {}

    for competitor in competitors:
        base_path = f"/data/clients/{client_id}/competitors/{competitor}"
        analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]
        if not analysis_dirs:
            continue

        analysis_dir = analysis_dirs[0]
        competitor_path = f"{base_path}/{analysis_dir}"

        # Load winning buckets
        with open(f"{competitor_path}/winner_analysis.json") as f:
            winner_data = json.load(f)

        winning_buckets = winner_data["top_3_buckets"]
        competitor_results = {}

        for bucket in winning_buckets:
            # Aggregate content patterns for this bucket (top performers only)
            aggregated = aggregate_content_classifications(
                bucket_name=bucket,
                base_path=competitor_path,
                performer_type="top"
            )

            if not aggregated:
                continue

            # Extract top N for each field
            bucket_results = {
                "top_2_categories": [c[0] for c in aggregated["content_category"].most_common(2)],
                "top_2_drivers": [d[0] for d in aggregated["engagement_drivers"].most_common(2)],
                "top_2_hooks": [h[0] for h in aggregated["hook_strategy"].most_common(2)],
                "top_2_cta_strategies": [c[0] for c in aggregated["closing_strategy"].most_common(2)],  # Video ending CTA
                "top_2_caption_ctas": [c[0] for c in aggregated["caption_cta_type"].most_common(2)],  # Caption CTA
                "top_3_pain_points": [p[0] for p in aggregated["pain_points"].most_common(3)],
                "top_3_keywords": [k[0] for k in aggregated["keywords"].most_common(3)],
                "top_2_tactics": [t[0] for t in aggregated["content_tactics"].most_common(2)]
            }

            competitor_results[bucket] = bucket_results

        results[competitor] = competitor_results

    return results
```

---

#### Function 6: `extract_hashtag_analysis()`

**Purpose**: Extract hashtag patterns from all winning buckets (per competitor)

**Implementation**: Same as Report 3 - see Section 3.3 Function 5

---

#### Function 7: `extract_mention_analysis()`

**Purpose**: Extract @mention patterns for content sourcing (per competitor)

**Implementation**: Same as Report 3 - see Section 3.3 Function 6

---

#### Function 8: `select_qr_code_videos()`

**Purpose**: Select top performer video for QR code generation (per competitor)

**Implementation**: Same as Report 3 - see Section 3.3 Function 7

---

#### Function 9: `generate_qr_codes()`

**Purpose**: Generate QR code PNG files from TikTok URLs

**Implementation**: Same as Report 3 - see Section 3.3 Function 8

---

#### Function 10: `extract_common_winning_buckets()`

**Purpose**: Identify common winning buckets across multiple competitors

**Used by**: Report 4 (Section 5: Supplementary Insights)

**Signature**:
```python
def extract_common_winning_buckets(client_id, competitor_handles, mode='top', strategy='contrastive'):
    """
    Find common buckets that appear in 2+ competitors' top_3_buckets

    Args:
        client_id: Client identifier
        competitor_handles: List of competitor handles (e.g., ['drinkpoppi', 'nike', 'vitalproteins'])
        mode: 'top' or 'bottom'
        strategy: 'contrastive' (default)

    Returns:
        dict: {
            'common_buckets': ['18-33s', '13-18s', ...],  # Sorted by frequency
            'bucket_competitors': {
                '18-33s': ['@drinkpoppi', '@nike', '@vitalproteins'],
                '13-18s': ['@drinkpoppi', '@vitalproteins'],
                ...
            }
        }
    """
    from collections import Counter
    import json
    import os

    bucket_frequency = Counter()
    competitor_buckets = {}

    # Load each competitor's top_3_buckets
    for handle in competitor_handles:
        base_path = f"/data/clients/{client_id}/competitors/{handle}/{mode}_{strategy}"
        winner_path = os.path.join(base_path, 'winner_analysis.json')

        if os.path.exists(winner_path):
            with open(winner_path, 'r') as f:
                winner_data = json.load(f)
                top_3 = winner_data.get('top_3_buckets', [])
                competitor_buckets[handle] = top_3

                for bucket in top_3:
                    bucket_frequency[bucket] += 1

    # Find common buckets (appear in 2+ competitors)
    common_buckets = [bucket for bucket, count in bucket_frequency.most_common() if count >= 2]

    # Map competitors to each common bucket
    bucket_competitors = {}
    for bucket in common_buckets:
        bucket_competitors[bucket] = [
            f"@{handle}" for handle, buckets in competitor_buckets.items()
            if bucket in buckets
        ]

    return {
        'common_buckets': common_buckets,
        'bucket_competitors': bucket_competitors
    }
```

**Example Usage**:
```python
result = extract_common_winning_buckets('acme', ['drinkpoppi', 'nike', 'vitalproteins'])
# Returns:
# {
#     'common_buckets': ['18-33s', '13-18s'],
#     'bucket_competitors': {
#         '18-33s': ['@drinkpoppi', '@nike', '@vitalproteins'],
#         '13-18s': ['@drinkpoppi', '@vitalproteins']
#     }
# }
```

---

#### Function 11: Inline Helper Functions

```python
def format_views(view_count):
    """Format view count with K or M suffix."""
    if view_count >= 1000000:
        return f"{view_count / 1000000:.1f}M"
    elif view_count >= 1000:
        return f"{int(view_count / 1000)}K"
    else:
        return str(view_count)


def calculate_posting_frequency(client_id, competitor_handle):
    """Calculate videos per week from winner_analysis.json."""
    import json
    import os

    base_path = f"/data/clients/{client_id}/competitors/{competitor_handle}"
    analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]

    if not analysis_dirs:
        raise FileNotFoundError(f"No analysis directory found")

    analysis_dir = analysis_dirs[0]
    winner_analysis_path = f"{base_path}/{analysis_dir}/winner_analysis.json"

    with open(winner_analysis_path) as f:
        data = json.load(f)

    total_videos = sum(data["top_100_distribution"].values())
    weeks = 90 / 7  # ~13 weeks

    posting_freq = round(total_videos / weeks, 1)

    return posting_freq


def calculate_bucket_distribution(winner_analysis_path):
    """Calculate percentage distribution across 8 buckets."""
    import json

    with open(winner_analysis_path) as f:
        data = json.load(f)

    bucket_distribution = data["bucket_distribution"]
    total_videos = sum(bucket_distribution.values())

    bucket_percentages = {
        bucket: round((count / total_videos) * 100)
        for bucket, count in bucket_distribution.items()
    }

    return bucket_percentages


# NOTE: aggregate_content_classifications() is defined in Section 0.5.1
# Use the updated signature:
#   aggregate_content_classifications(bucket_name, base_path, performer_type="top")
```

---

### Data Source File Formats

Same data sources as Reports 1-3:
1. `winner_analysis.json` - See Section 3.3 File 1
2. `selected_videos.json` (per bucket) - See Section 3.2 File 4
3. `content_analysis/validated/bucket_{name}/{video_id}_content.json` (per-bucket organization) - See Section 0.2

---

### Complete Implementation Pattern

```python
#!/usr/bin/env python3
"""
extract_multi_competitor_data.py - Report 4: Multi-Competitor Market Intelligence

Generates market intelligence report comparing 2-5 competitors.

Usage:
    python extract_multi_competitor_data.py --client acme --competitors drinkpoppi,nike,vitalproteins
"""

import argparse
import json
import os
import pandas as pd
from collections import Counter

# Import functions defined above


def main():
    """Main extraction workflow"""

    # =============================
    # STEP 1: Parse CLI Arguments
    # =============================
    parser = argparse.ArgumentParser(description='Extract Report 4: Multi-Competitor Market Intelligence')
    parser.add_argument('--client', required=True, help='Client ID (e.g., acme)')
    parser.add_argument('--competitors', required=True, help='Comma-separated competitor handles (e.g., drinkpoppi,nike,vitalproteins)')
    parser.add_argument('--mode', default='top', help='Mode (default: top)')
    parser.add_argument('--strategy', default='contrastive', help='Strategy (default: contrastive)')
    args = parser.parse_args()

    competitors = [c.strip() for c in args.competitors.split(',')]

    print(f"\nRunning multi-competitor extraction")
    print(f"Analyzing {len(competitors)} competitors: {', '.join('@' + c for c in competitors)}")

    # =============================
    # STEP 2: Build Output Path
    # =============================
    # Note: This assumes all competitors are analyzed within the same hashtag context
    # For cross-hashtag analysis, adjust the base path accordingly
    output_base = f"/data/clients/{args.client}/market_intelligence/multi_competitor/reports/multicompetitor"
    os.makedirs(output_base, exist_ok=True)

    # =============================
    # STEP 3: Calculate Performance Rankings
    # =============================
    print("Loading performance data for all competitors...")
    rankings = calculate_competitor_rankings(args.client, competitors)

    market_leader = rankings[0] if rankings else None

    # =============================
    # STEP 4: Build Bucket Distribution Matrix
    # =============================
    print(f"Building bucket distribution matrix (8 buckets × {len(competitors)} competitors)...")
    bucket_matrix = build_bucket_distribution_matrix(args.client, competitors)

    # =============================
    # STEP 5: Build Performance Matrix
    # =============================
    performance_matrix = build_performance_matrix(args.client, competitors)
    unique_buckets = performance_matrix["unique_buckets"]
    print(f"Building performance matrix ({len(unique_buckets)} unique winning buckets × {len(competitors)} competitors)...")

    # =============================
    # STEP 6: Calculate Posting Frequency
    # =============================
    posting_freqs = []
    for competitor in competitors:
        freq = calculate_posting_frequency(args.client, competitor)
        posting_freqs.append(freq)

    market_avg_freq = round(sum(posting_freqs) / len(posting_freqs), 1)

    # =============================
    # STEP 7: Aggregate Per-Bucket Content Intelligence
    # =============================
    total_combinations = sum(
        len([1 for c in competitors if c in bucket_content_data])
        for bucket_content_data in [{}] * 3  # Approximate
    )
    print(f"Aggregating per-bucket content intelligence ({total_combinations}+ bucket-competitor combinations)...")

    per_bucket_content = aggregate_per_bucket_content(args.client, competitors)

    # =============================
    # STEP 8: Extract Hashtag & Mention Analysis (Per Competitor)
    # =============================
    print(f"Extracting hashtag and mention analysis for {len(competitors)} competitors...")

    hashtag_data = {}
    mention_data = {}

    for competitor in competitors:
        hashtag_data[competitor] = extract_hashtag_analysis(args.client, competitor)
        mention_data[competitor] = extract_mention_analysis(args.client, competitor)

    # =============================
    # STEP 9: Select QR Code Videos & Generate
    # =============================
    print(f"Generating {len(competitors)} QR codes...")

    qr_output_dir = os.path.join(output_base, 'qr_codes')
    qr_data_list = []
    qr_metadata = {}

    for competitor in competitors:
        # Get top performer from best bucket
        base_path = f"/data/clients/{args.client}/competitors/{competitor}"
        analysis_dirs = [d for d in os.listdir(base_path) if d.startswith('top_')]
        if not analysis_dirs:
            continue

        analysis_dir = analysis_dirs[0]
        competitor_path = f"{base_path}/{analysis_dir}"

        with open(f"{competitor_path}/winner_analysis.json") as f:
            winner_data = json.load(f)

        best_bucket = winner_data["top_3_buckets"][0]  # First winning bucket
        best_bucket_path = f"{competitor_path}/buckets/bucket_{best_bucket}"

        qr_video = select_qr_code_videos(best_bucket_path, "top")

        # Collect metadata
        qr_metadata[competitor] = {
            "file": f"{competitor}_top.png",
            "url": qr_video["url"],
            "views": qr_video["views"],
            "engagement": qr_video["engagement"],
            "duration": qr_video["duration"],
            "bucket": best_bucket,
            "hashtags": " ".join([f"#{h['name']}" for h in qr_video.get("hashtags", [])[:4]])
        }

        qr_data_list.append({
            "filename": f"{competitor}_top.png",
            "url": qr_video["url"]
        })

    generate_qr_codes(qr_data_list, qr_output_dir)

    # =============================
    # STEP 10: Build Excel Data Structure
    # =============================
    tab_data = []

    # PAGE 1: MARKET OVERVIEW
    tab_data.append(['PAGE_1_MARKET_OVERVIEW', ''])
    tab_data.append(['', ''])

    tab_data.append(['COMPETITOR_COUNT', str(len(competitors))])
    for i, competitor in enumerate(competitors, 1):
        tab_data.append([f'COMPETITOR_{i}_HANDLE', f'@{competitor}'])
    tab_data.append(['ANALYSIS_PERIOD', 'Last 90 days'])
    tab_data.append(['', ''])

    # Performance Rankings
    for ranking in rankings:
        rank = ranking["rank"]
        tab_data.append([f'RANK_{rank}_HANDLE', ranking["handle"]])
        tab_data.append([f'RANK_{rank}_AVG_VIEWS', format_views(ranking["avg_views"])])
        tab_data.append([f'RANK_{rank}_AVG_ENGAGEMENT', str(ranking["avg_engagement"])])
        tab_data.append([f'RANK_{rank}_POSTING_FREQ', str(ranking["posting_freq"])])
        tab_data.append([f'RANK_{rank}_VIDEOS_ANALYZED', str(ranking["videos_analyzed"])])
        tab_data.append(['', ''])

    if market_leader:
        tab_data.append(['MARKET_LEADER', market_leader["handle"]])
        tab_data.append(['MARKET_LEADER_REASON', f"{format_views(market_leader['avg_views'])} avg views, {market_leader['avg_engagement']}% engagement, highest posting frequency"])

    # Analysis Scope
    tab_data.append(['', ''])
    for i, ranking in enumerate(rankings, 1):
        tab_data.append([f'COMP_{i}_VIDEOS_ANALYZED', str(ranking["videos_analyzed"])])

    # PAGE 2: CONTENT STRATEGY COMPARISON
    tab_data.append(['', ''])
    tab_data.append(['PAGE_2_CONTENT_STRATEGY', ''])
    tab_data.append(['', ''])

    # Bucket Distribution Matrix
    for bucket in bucket_matrix["buckets"]:
        bucket_data = bucket_matrix["matrix"][bucket]
        bucket_key = bucket.replace("-", "_").upper()

        for i, pct in enumerate(bucket_data["competitors"], 1):
            tab_data.append([f'BUCKET_{bucket_key}_COMP_{i}_PCT', str(pct)])

            # High volume marker
            if bucket_data["high_volume_markers"][i-1]:
                tab_data.append([f'BUCKET_{bucket_key}_COMP_{i}_HIGH_VOLUME', 'True'])

        tab_data.append([f'BUCKET_{bucket_key}_MARKET_PATTERN', bucket_data["market_pattern"]])
        tab_data.append(['', ''])

    # Performance Matrix
    tab_data.append(['UNIQUE_WINNING_BUCKETS_COUNT', str(len(unique_buckets))])
    for i, bucket in enumerate(unique_buckets, 1):
        tab_data.append([f'UNIQUE_WINNING_BUCKET_{i}', bucket])
    tab_data.append(['', ''])

    for bucket in unique_buckets:
        bucket_key = bucket.replace("-", "_").upper()
        bucket_data = performance_matrix["matrix"][bucket]

        for i, comp_data in enumerate(bucket_data["competitors"], 1):
            prefix = f'PERF_{bucket_key}'

            if comp_data["views"] is not None:
                tab_data.append([f'{prefix}_VIEWS_COMP_{i}', format_views(comp_data["views"])])
                tab_data.append([f'{prefix}_ENGAGEMENT_COMP_{i}', str(comp_data["engagement"])])
            else:
                tab_data.append([f'{prefix}_VIEWS_COMP_{i}', '—'])
                tab_data.append([f'{prefix}_ENGAGEMENT_COMP_{i}', '—'])

            tab_data.append([f'{prefix}_WINNING_COMP_{i}', str(comp_data["is_winning"])])

        tab_data.append([f'PERF_{bucket_key}_BEST_PERFORMER', bucket_data["best_performer"]])
        tab_data.append(['', ''])

    # Posting Frequency
    for i, freq in enumerate(posting_freqs, 1):
        tab_data.append([f'POSTING_FREQ_COMP_{i}', str(freq)])
    tab_data.append(['MARKET_AVG_POSTING_FREQ', str(market_avg_freq)])

    # PAGE 3: CREATIVE INTELLIGENCE (PER-BUCKET)
    tab_data.append(['', ''])
    tab_data.append(['PAGE_3_CREATIVE_INTELLIGENCE', ''])
    tab_data.append(['', ''])

    # Per-bucket content intelligence
    for i, competitor in enumerate(competitors, 1):
        if competitor not in per_bucket_content:
            continue

        competitor_buckets = per_bucket_content[competitor]

        for bucket, bucket_data in competitor_buckets.items():
            bucket_key = bucket.replace("-", "_").upper()

            # Content categories (top 2)
            for j, category in enumerate(bucket_data["top_2_categories"], 1):
                tab_data.append([f'CONTENT_CAT_COMP_{i}_BUCKET_{bucket_key}_{j}', category.replace('_', ' ').title()])

            # Engagement drivers (top 2)
            for j, driver in enumerate(bucket_data["top_2_drivers"], 1):
                tab_data.append([f'ENGAGEMENT_DRIVER_COMP_{i}_BUCKET_{bucket_key}_{j}', driver.replace('_', ' ').title()])

            # Hook strategies (top 2)
            for j, hook in enumerate(bucket_data["top_2_hooks"], 1):
                tab_data.append([f'HOOK_STRATEGY_COMP_{i}_BUCKET_{bucket_key}_{j}', hook.replace('_', ' ').title()])

            # CTA strategies (top 2)
            for j, cta in enumerate(bucket_data["top_2_ctas"], 1):
                tab_data.append([f'CTA_STRATEGY_COMP_{i}_BUCKET_{bucket_key}_{j}', cta.replace('_', ' ').title()])

            # Pain points (top 3)
            for j, pain in enumerate(bucket_data["top_3_pain_points"], 1):
                tab_data.append([f'PAIN_POINT_COMP_{i}_BUCKET_{bucket_key}_{j}', pain.replace('_', ' ').title()])

            # Keywords (top 3)
            for j, keyword in enumerate(bucket_data["top_3_keywords"], 1):
                tab_data.append([f'KEYWORD_COMP_{i}_BUCKET_{bucket_key}_{j}', keyword])

            # Content tactics (top 2)
            for j, tactic in enumerate(bucket_data["top_2_tactics"], 1):
                tab_data.append([f'CONTENT_TACTIC_COMP_{i}_BUCKET_{bucket_key}_{j}', tactic.replace('_', ' ').title()])

            tab_data.append(['', ''])

    # Supplementary Insights (per-bucket, per-competitor)
    tab_data.append(['', ''])

    # Step 1: Find common winning buckets across competitors
    common_bucket_data = extract_common_winning_buckets(args.client, competitors)
    common_buckets = common_bucket_data['common_buckets']
    bucket_competitors_map = common_bucket_data['bucket_competitors']

    # Step 2: For each common bucket, extract insights for applicable competitors
    for bucket_name in common_buckets:
        applicable_competitors = bucket_competitors_map[bucket_name]

        for competitor_handle in applicable_competitors:
            # Find competitor's global index (1-based)
            competitor_clean = competitor_handle.replace('@', '')
            comp_global_idx = competitors.index(competitor_clean) + 1

            # Load winning_formulas.json for this competitor + bucket
            base_path = f"/data/clients/{args.client}/competitors/{competitor_clean}/{args.mode}_{args.strategy}"
            bucket_path = os.path.join(base_path, 'buckets', f'bucket_{bucket_name}')
            formulas_path = os.path.join(bucket_path, 'ml_analysis', 'llm', 'winning_formulas.json')

            if os.path.exists(formulas_path):
                with open(formulas_path, 'r') as f:
                    formulas = json.load(f)
                    insights = formulas.get('supplementary_insights', {}).get('universal_principles', [])

                    # Extract top 5 insights
                    bucket_key = bucket_name.replace('-', '_').upper()  # e.g., "18_33S"
                    for i in range(min(5, len(insights))):
                        field_name = f'SUPP_INSIGHT_COMP_{comp_global_idx}_BUCKET_{bucket_key}_{i+1}'
                        tab_data.append([field_name, insights[i]])

            # Add empty row between competitors within same bucket
            tab_data.append(['', ''])

        # Add empty row between buckets
        tab_data.append(['', ''])

    # Hashtag Strategy (aggregate per competitor)
    tab_data.append(['', ''])
    for i, competitor in enumerate(competitors, 1):
        if competitor not in hashtag_data:
            continue

        h_data = hashtag_data[competitor]

        tab_data.append([f'HASHTAG_TOTAL_UNIQUE_COMP_{i}', str(h_data["total_unique_hashtags"])])
        tab_data.append([f'HASHTAG_AVG_PER_VIDEO_COMP_{i}', str(int(h_data["avg_hashtags_per_video"]))])
        tab_data.append([f'HASHTAG_TOP_5_CONCENTRATION_COMP_{i}', str(h_data["top_5_concentration"])])

        strategy_type = "Diversified" if h_data["total_unique_hashtags"] > 20 else "Focused"
        tab_data.append([f'HASHTAG_STRATEGY_TYPE_COMP_{i}', strategy_type])

        # Top 5 hashtags
        for j, h in enumerate(h_data["top_10_hashtags"][:5], 1):
            tab_data.append([f'HASHTAG_COMP_{i}_{j}', h["tag"]])
            tab_data.append([f'HASHTAG_COMP_{i}_{j}_PCT', str(h["usage_pct"])])

        tab_data.append(['', ''])

    # Caption Strategy (aggregate per competitor)
    for i, competitor in enumerate(competitors, 1):
        if competitor not in hashtag_data:
            continue

        h_data = hashtag_data[competitor]
        tab_data.append([f'CAPTION_AVG_HASHTAG_COUNT_COMP_{i}', str(int(h_data["avg_hashtags_per_video"]))])

        # Top CTA (would come from aggregated content classifications)
        tab_data.append([f'CAPTION_TOP_CTA_COMP_{i}', 'Link in bio'])  # Placeholder
        tab_data.append([f'CAPTION_TOP_CTA_COMP_{i}_PCT', '68'])  # Placeholder

    # Content Sourcing Strategy (per competitor)
    tab_data.append(['', ''])
    for i, competitor in enumerate(competitors, 1):
        if competitor not in mention_data:
            continue

        m_data = mention_data[competitor]

        tab_data.append([f'SOURCING_UGC_PCT_COMP_{i}', str(m_data["repost_rate"])])
        tab_data.append([f'SOURCING_OWN_PCT_COMP_{i}', str(100 - m_data["repost_rate"])])
        tab_data.append([f'SOURCING_UNIQUE_AFFILIATES_COMP_{i}', str(m_data["total_unique_mentions"])])

        # Top 4 affiliates
        for j, affiliate in enumerate(m_data["top_10_mentions"][:4], 1):
            tab_data.append([f'AFFILIATE_COMP_{i}_{j}_HANDLE', affiliate["handle"]])
            tab_data.append([f'AFFILIATE_COMP_{i}_{j}_PCT', str(affiliate["percentage"])])
            tab_data.append([f'AFFILIATE_COMP_{i}_{j}_COUNT', str(affiliate["mention_count"])])

        tab_data.append(['', ''])

    # PAGE 4: VISUAL EXAMPLES (QR CODES)
    tab_data.append(['', ''])
    tab_data.append(['PAGE_4_VISUAL_EXAMPLES', ''])
    tab_data.append(['', ''])

    for i, competitor in enumerate(competitors, 1):
        if competitor not in qr_metadata:
            continue

        qr_meta = qr_metadata[competitor]

        tab_data.append([f'QR_COMP_{i}_FILE', qr_meta["file"]])
        tab_data.append([f'QR_COMP_{i}_URL', qr_meta["url"]])
        tab_data.append([f'QR_COMP_{i}_VIEWS', format_views(qr_meta["views"])])
        tab_data.append([f'QR_COMP_{i}_ENGAGEMENT', str(qr_meta["engagement"])])
        tab_data.append([f'QR_COMP_{i}_DURATION', f"{qr_meta['duration']}s"])
        tab_data.append([f'QR_COMP_{i}_BUCKET', qr_meta["bucket"]])
        tab_data.append([f'QR_COMP_{i}_HASHTAGS', qr_meta["hashtags"]])
        tab_data.append(['', ''])

    # =============================
    # STEP 11: Write Excel File
    # =============================
    excel_filename = "market_intelligence_report.xlsx"
    excel_path = os.path.join(output_base, excel_filename)

    df = pd.DataFrame(tab_data, columns=['Field Name', 'Value'])
    df.to_excel(excel_path, sheet_name='Report_Data', index=False, engine='openpyxl')

    # =============================
    # STEP 12: Print Success Message
    # =============================
    print(f"\n✓ Extraction complete")
    print(f"  Excel: {excel_path}")
    print(f"  QR codes: {qr_output_dir} ({len(qr_data_list)} files)")
    print(f"  Total fields: {len(tab_data)}")


if __name__ == '__main__':
    main()
```

---

### Implementation Notes for Developer

**TODO items in skeleton above**:
1. ✅ All core multi-competitor functions implemented
2. ✅ Matrix building functions complete
3. ✅ Per-bucket aggregation logic implemented
4. ✅ QR code generation for multiple competitors
5. ⚠️ Caption CTA aggregation needs implementation (currently placeholder)
6. ⚠️ Taxonomy descriptions for categories/hooks/drivers (same as Report 3)

**Testing checklist**:
- [ ] Script runs without errors with 2, 3, 4, and 5 competitors
- [ ] Excel file created with single tab
- [ ] All fields populated (field count scales correctly with competitor count)
- [ ] Bucket distribution matrix complete (8 × N)
- [ ] Performance matrix shows only winning buckets per competitor
- [ ] Per-bucket content intelligence aggregates correctly
- [ ] High volume markers accurate (>20% threshold)
- [ ] Market patterns calculated correctly
- [ ] Best performer identified per bucket (ties broken by engagement)
- [ ] N QR codes generated (1 per competitor)
- [ ] QR codes scan correctly to TikTok video URLs
- [ ] Hashtag and mention analysis accurate for all competitors

**Error handling**:
Script should exit with clear error if:
- No competitors specified or invalid format
- Fewer than 2 or more than 5 competitors specified
- `winner_analysis.json` not found for any competitor
- `selected_videos.json` not found for any winning bucket
- JSON files malformed
- Cannot write Excel file (permissions)
- Cannot create output directories
- QR code generation fails

**Dependencies**:
```bash
pip install pandas openpyxl qrcode[pil]
```

**Performance Considerations**:
- 5 competitors with 3 winning buckets each = 15 bucket analyses
- Per-bucket content aggregation for 15 buckets can be slow
- Consider parallel processing for production implementation
- Estimated runtime: 30-60 seconds for 3 competitors

---

**END OF SECTION 3.4**

This section is complete and self-contained for implementation of `extract_multi_competitor_data.py`.
