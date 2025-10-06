

## Client Architecture & Storage

### Data Example

#### ML Production Pipeline Architecture
```
/data/
├── clients/
│   ├── {client_id}/
│   │   ├── hashtags/                       # Hashtag-based analyses
│   │   │   └── {hashtag_name}/             # e.g., "nutrition", "fitness"
│   │   │       ├── top_contrastive/        # Analysis: top mode + contrastive strategy
│   │   │       │   ├── config.json         # {mode: "top", strategy: "contrastive", date_filter: "last_90_days", run_date: "2025-01-28", video_count: 300}
│   │   │       │   ├── buckets/            # Duration-based ML buckets (8 total)
│   │   │       │   │   ├── bucket_0-3s/
│   │   │       │   │   │   ├── videos/     # Raw MP4s: N files (configurable via --video-count)
│   │   │       │   │   │   │                # Contrastive: 80% top + 20% bottom of N (e.g., 80+20 if N=100)
│   │   │       │   │   │   ├── analysis/   # RumiAI outputs for N videos
│   │   │       │   │   │   │   ├── insights/   # temporal_windows JSON (1 per video)
│   │   │       │   │   │   │   ├── unified/    # Intermediate timeline+ml_data (debugging)
│   │   │       │   │   │   │   └── service_debug/  # emotion_detection, audio_energy outputs
│   │   │       │   │   │   ├── ml_analysis/    # ML pipeline outputs
│   │   │       │   │   │   │   ├── aggregated_features.csv          # Aggregated temporal windows (N videos)
│   │   │       │   │   │   │   ├── rf_transformed.csv               # RF-ready features
│   │   │       │   │   │   │   ├── km_transformed.csv               # KMeans-ready features
│   │   │       │   │   │   │   ├── random_forest_analysis.json      # ~30KB - Input to LLM Call 1
│   │   │       │   │   │   │   └── kmeans_analysis.json             # ~30KB - Input to LLM Call 1
│   │   │       │   │   │   ├── models/     # Trained models for THIS bucket
│   │   │       │   │   │   │   ├── random_forest_v1.pkl  # Classification model
│   │   │       │   │   │   │   ├── kmeans_v1.pkl         # Clustering model
│   │   │       │   │   │   │   ├── scalers.pkl            # MinMaxScalers for KMeans
│   │   │       │   │   │   │   └── model_metrics.json
│   │   │       │   │   │   ├── llm_reports/  # LLM outputs
│   │   │       │   │   │   │   ├── analysis/              # LLM Call 1 outputs (insight extraction)
│   │   │       │   │   │   │   │   ├── call_1_rf_prompt.txt
│   │   │       │   │   │   │   │   ├── call_1_rf_raw_response.json
│   │   │       │   │   │   │   │   ├── call_1_kmeans_prompt.txt
│   │   │       │   │   │   │   │   ├── call_1_kmeans_raw_response.json
│   │   │       │   │   │   │   │   └── insights.json              # Structured insights (parsed) → Input to LLM Call 2
│   │   │       │   │   │   │   └── formatted/            # LLM Call 2 outputs (report generation)
│   │   │       │   │   │   │       ├── call_2_prompt.txt
│   │   │       │   │   │   │       ├── call_2_raw_response.json
│   │   │       │   │   │   │       ├── rf_feature_importance.md
│   │   │       │   │   │   │       ├── strategy_1_the_educator.md
│   │   │       │   │   │   │       ├── strategy_2_visual_storyteller.md
│   │   │       │   │   │   │       ├── strategy_3_personal_journey.md
│   │   │       │   │   │   │       └── bucket_summary.md
│   │   │       │   │   │   ├── reports/    # Final PDFs (no LLM - converted from markdown)
│   │   │       │   │   │   ├── checkpoints/ # Processing state for this bucket
│   │   │       │   │   │   └── logs/       # Bucket processing logs
│   │   │       │   │   ├── bucket_3-9s/
│   │   │       │   │   ├── bucket_9-13s/
│   │   │       │   │   ├── bucket_13-18s/
│   │   │       │   │   ├── bucket_18-33s/
│   │   │       │   │   ├── bucket_33-60s/
│   │   │       │   │   ├── bucket_60-90s/
│   │   │       │   │   └── bucket_90-120s/
│   │   │       │   └── hashtag_summary/   # Cross-bucket executive reports
│   │   │       │       ├── executive_report.pdf
│   │   │       │       └── hashtag_metrics.json
│   │   │       │
│   │   │       ├── top_top/               # Analysis: top mode + top strategy (OPTIONAL)
│   │   │       │   ├── config.json        # {mode: "top", strategy: "top", date_filter: "last_90_days", run_date: "2025-02-01", video_count: 300}
│   │   │       │   ├── buckets/           # Same structure
│   │   │       │   │   └── bucket_0-3s/
│   │   │       │   │       ├── videos/    # N files (top only, N from --video-count, default 40)
│   │   │       │   │       ├── analysis/  # N JSONs
│   │   │       │   │       └── reports/   # Best practices reports (no classification models)
│   │   │       │   └── hashtag_summary/
│   │   │
│   │   ├── competitors/                   # Competitor tracking (what rivals are doing)
│   │   │   └── {competitor_handle}/       # e.g., "@rival_brand"
│   │   │       ├── top_top/               # Analysis: top mode + top strategy
│   │   │       │   ├── config.json        # {mode: "top", strategy: "top", date_filter: "last_90_days", run_date: "2025-01-28", video_count: 150}
│   │   │       │   ├── buckets/           # Same 8-bucket structure
│   │   │       │   │   └── bucket_0-3s/
│   │   │       │   │       ├── videos/    # N files (top only, N from --video-count, default 40)
│   │   │       │   │       ├── analysis/
│   │   │       │   │       └── reports/   # Best practices reports
│   │   │       │   └── competitor_summary/
│   │   │       │       ├── competitor_report.pdf
│   │   │       │       └── competitor_metrics.json
│   │   │       │
│   │   │       ├── recent_top/            # Analysis: recent mode + top strategy (OPTIONAL)
│   │   │       │   ├── config.json        # Track current strategy shifts
│   │   │       │   └── buckets/
│   │   │       │
│   │   │       └── top_contrastive/       # Analysis: top mode + contrastive strategy (OPTIONAL)
│   │   │           ├── config.json        # Full contrastive competitor analysis
│   │   │           └── buckets/
│   │   │
│   │   └── creators/                      # Creator vetting (potential affiliates)
│   │       └── {creator_handle}/          # e.g., "@potential_affiliate"
│   │           ├── recent_top/            # Analysis: recent mode + top strategy
│   │           │   ├── config.json        # {mode: "recent", strategy: "top", date_filter: "last_30_days", run_date: "2025-01-28", video_count: 40}
│   │           │   ├── buckets/           # Same 8-bucket structure
│   │           │   │   └── bucket_0-3s/
│   │           │   │       ├── videos/    # N files (top recent videos, N from --video-count, default 40)
│   │           │   │       └── analysis/
│   │           │   └── creator_summary/
│   │           │       ├── style_profile.pdf
│   │           │       ├── compatibility_report.pdf
│   │           │       ├── compatibility_scores.json
│   │           │       └── creator_metrics.json
│   │           │
│   │           └── top_top/               # Analysis: top mode + top strategy (OPTIONAL)
│   │               ├── config.json        # Peak performance analysis
│   │               └── buckets/
```

**Architecture Notes**:
- **Analysis Directories**: Each `{mode}_{strategy}/` directory is a complete, independent analysis run
- **config.json**: Stores run parameters (mode, strategy, date_filter, run_date, video_count) for reproducibility
- **Coexistence**: Multiple analyses can exist simultaneously without overwriting
- **Default Paths**: Each flow type writes to its default analysis directory (hashtag→top_contrastive, competitor→top_top, creator→recent_top)
- **Video Counts**: User-configurable via --video-count N
  - Contrastive: N per qualified bucket (80/20 split, default N=100)
  - Top: N per qualified bucket (all top, default N=40)
  - Only top 3 most active buckets are processed (adaptive bucket processing)




### Data Retention Policy
- **Raw Videos**: 30 days (then delete to save space, can re-download if needed)
- **ML Analysis**: 6 months (compressed after 30 days)
- **ML Models**: Keep latest 3 versions per client/hashtag
- **Reports**: Indefinite (small size, high value)
- **Checkpoints**: 7 days after successful completion

### Storage Cost Optimization
- **Video Deletion**: Remove raw videos after 30 days

---

## ML Analysis Pipeline

This section documents the end-to-end ML pipeline from video processing to LLM-generated creative reports.

### Pipeline Overview

```
temporal_windows_updated.json (N videos per qualified bucket)
    ↓
    aggregation
    ↓
aggregated_features.csv (N rows × ~35 columns)
    ↓
    transformation (FeatureTransformation.md spec)
    ↓
rf_transformed.csv + km_transformed.csv
    ↓
    ML training
    ↓
random_forest_analysis.json + kmeans_analysis.json (~30KB each)
    ↓
    LLM Call 1: Analysis (extract insights)
    ↓
insights.json (structured ML insights)
    ↓
    LLM Call 2: Formatting (generate reports)
    ↓
5 markdown reports
    ↓
    PDF generation (no LLM)
    ↓
5 PDF creative strategy reports per bucket (40 total)
```

---

### Stage 1: Feature Aggregation

**Input**: N × `temporal_windows_updated.json` files per qualified bucket (N from --video-count)
- Each JSON contains features per temporal window (hook, middle segments, closing)
- Middle segment count varies by video duration (2-7 windows total)

**Process**:
- Aggregate temporal windows to video level:
  - Hook features: Use directly (always 1 hook window)
  - Middle features: Average across all middle segments (handles variable count)
  - Closing features: Use directly (always 1 closing window)
  - Global features: Sum or derive from all windows

**Output**: `ml_analysis/aggregated_features.csv`
- Shape: (N videos, ~35 aggregated features)
- Example columns: `hook_scene_count`, `middle_avg_word_count`, `closing_energy_level`, `duration`

**Reference**: See [FeatureTransformation.md](./FeatureTransformation.md) "Temporal Features to ML Training Input" section

---

### Stage 2: Feature Transformation

**Input**: `ml_analysis/aggregated_features.csv`

**Process**:
- **RF Transformation**:
  - Apply one-hot encoding to categorical features
  - Extract temporal features from `create_time` (hour, day, month, weekend, business_hours)
  - Direct use of numerical features (scale-invariant)

- **KMeans Transformation**:
  - Apply log + scale to right-skewed features (counts, variances)
  - Scale [0-1] for already-normalized features
  - Cyclical encoding for `create_time` (sin/cos pairs)
  - One-hot encoding for `dominant_emotion_id`

**Outputs**:
- `ml_analysis/rf_transformed.csv` (N videos, ~39 features)
- `ml_analysis/km_transformed.csv` (N videos, ~40 features)

**Reference**: See [FeatureTransformation.md](./FeatureTransformation.md) for complete transformation specifications

---

### Stage 3: ML Model Training

**Input**:
- `ml_analysis/rf_transformed.csv`
- `ml_analysis/km_transformed.csv`

**Process**:

**Random Forest Training**:
```python
# Classification: Top 80% vs Bottom 20% performers
# Example: If N=100, top 80 vs bottom 20
# Example: If N=150, top 120 vs bottom 30
X = rf_transformed  # (N, 39)
y = is_top_performer  # (N,) - binary labels

rf_model = RandomForest(n_estimators=100).fit(X, y)
feature_importance = rf_model.feature_importances_
predictions = rf_model.predict_proba(X)
```

**K-Means Training**:
```python
# Clustering: Identify creative patterns
X = km_transformed  # (N, 40)

# Fit scalers per bucket (save for inference)
scalers = {}
for feature in X.columns:
    scalers[feature] = MinMaxScaler().fit(X[[feature]])
X_scaled = apply_scalers(X, scalers)

kmeans_model = KMeans(n_clusters=3).fit(X_scaled)
cluster_assignments = kmeans_model.labels_
cluster_centroids = kmeans_model.cluster_centers_
```

**Outputs**:
- `models/random_forest_v1.pkl`
- `models/kmeans_v1.pkl`
- `models/scalers.pkl` (for K-Means inference)
- `models/model_metrics.json`

**Reference**: See [Kmeans.md](./Kmeans.md) for scaler fitting details

---

### Stage 4: LLM Input Generation

**Input**:
- `ml_analysis/aggregated_features.csv` (raw features)
- Trained model outputs (predictions, clusters, feature importance)

**Process**: Create structured JSONs for LLM analysis

**Random Forest JSON** (`ml_analysis/random_forest_analysis.json`):
```json
{
  "analysis_type": "random_forest",
  "bucket": "18-33s",
  "hashtag": "#nutrition",
  "video_count": 60,

  "feature_importance": [
    {"feature": "hook_eye_contact_rate", "importance": 0.22},
    {"feature": "middle_avg_word_count", "importance": 0.18},
    ...
  ],

  "videos": [
    {
      "video_id": "123",
      "is_top_performer": 1,
      "prediction_confidence": 0.92,
      "features": {
        "hook_scene_count": 3,
        "middle_avg_word_count": 55,
        ...
      }
    },
    ...
  ]
}
```

**K-Means JSON** (`ml_analysis/kmeans_analysis.json`):
```json
{
  "analysis_type": "kmeans",
  "bucket": "18-33s",
  "hashtag": "#nutrition",
  "n_clusters": 3,

  "cluster_summary": [
    {
      "cluster_id": 0,
      "cluster_name": "The Educator Pattern",
      "video_count": 22,
      "avg_engagement": 125000,
      "defining_features": {...}
    },
    ...
  ],

  "videos": [
    {
      "video_id": "123",
      "cluster_id": 0,
      "distance_to_centroid": 0.12,
      "features": {...}
    },
    ...
  ]
}
```

**Output Size**: ~30KB per JSON (2 JSONs per bucket = ~60KB)

**Reference**: See [ML_LLMData.md](./ML_LLMData.md) for full schema specifications

---

### Stage 5A: Analysis LLM Call (Insight Extraction)

**Purpose**: Extract structured insights from ML analysis without formatting concerns

**Input**:
- `ml_analysis/random_forest_analysis.json` (~30KB)
- `ml_analysis/kmeans_analysis.json` (~30KB)

**Process**:

**RF Analysis Call**:
```python
prompt = f"""
You are an ML analysis expert. Analyze Random Forest feature importance data.

HASHTAG: {hashtag}
BUCKET: {bucket}

INPUT DATA:
{rf_analysis_json}

TASK:
Extract insights in JSON format:

{{
  "top_features": [
    {{
      "feature": "hook_eye_contact_rate",
      "importance": 0.22,
      "interpretation": "Why this feature matters",
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "actionable_advice": "Specific action creators can take"
    }},
    ... (top 5 features)
  ],
  "improvement_opportunities": [
    {{
      "feature": "middle_avg_scene_count",
      "importance": 0.12,
      "current_gap": -1.5,
      "recommendation": "Increase middle pacing from 2.5 to 4.0 cuts"
    }},
    ... (top 3 opportunities)
  ],
  "model_performance": {{
    "accuracy": 0.88,
    "top_feature_importance": 0.22
  }}
}}

Focus on INSIGHTS, not formatting. Be precise with numbers.
"""

rf_insights = claude_api.generate(prompt, data=rf_analysis_json)
save("llm_reports/analysis/call_1_rf_raw_response.json", rf_insights)
```

**K-Means Analysis Call**:
```python
prompt = f"""
You are an ML analysis expert. Analyze K-Means clustering results.

HASHTAG: {hashtag}
BUCKET: {bucket}

INPUT DATA:
{kmeans_analysis_json}

TASK:
Extract insights in JSON format:

{{
  "clusters": [
    {{
      "cluster_id": 0,
      "name": "The Educator",
      "video_count": 22,
      "avg_engagement": 125000,
      "top_performer_percentage": 0.68,
      "defining_features": {{
        "hook": {{"high_eye_contact": 0.85, "moderate_scene_count": 3.2}},
        "middle": {{"high_word_count": 55, "consistent_emotion": 0.8}},
        "closing": {{"high_energy": 0.8}}
      }},
      "strategy_summary": "2-3 sentence description",
      "example_video_ids": ["123", "456", "789"]
    }},
    ... (3 clusters)
  ],
  "recommendations": {{
    "dominant_pattern": "The Educator (35% of top performers)",
    "best_for_beginners": "The Educator",
    "reasoning": "Why this pattern works best"
  }}
}}

Focus on INSIGHTS, not formatting. Be precise with numbers.
"""

kmeans_insights = claude_api.generate(prompt, data=kmeans_analysis_json)
save("llm_reports/analysis/call_1_kmeans_raw_response.json", kmeans_insights)
```

**Output Consolidation**:
```python
# Parse JSON responses
rf_data = json.loads(rf_insights)
kmeans_data = json.loads(kmeans_insights)

# Combine into single insights file
insights = {
    "bucket": bucket,
    "hashtag": hashtag,
    "rf_insights": rf_data,
    "kmeans_insights": kmeans_data,
    "generated_at": timestamp
}

save("llm_reports/analysis/insights.json", insights)
```

**Outputs**:
- `llm_reports/analysis/call_1_rf_prompt.txt`
- `llm_reports/analysis/call_1_rf_raw_response.json`
- `llm_reports/analysis/call_1_kmeans_prompt.txt`
- `llm_reports/analysis/call_1_kmeans_raw_response.json`
- `llm_reports/analysis/insights.json` (consolidated, structured)

**LLM Calls per Hashtag**: 16 total (2 per bucket × 8 buckets)

**Why Separate RF and K-Means Calls**:
- Smaller context per call (30KB vs 60KB)
- More focused prompts
- Easier to debug if one analysis fails
- Can run in parallel

**Why JSON Output**:
- Structured data (easy to parse)
- No formatting ambiguity
- Can validate schema
- Reusable for other outputs (dashboards, APIs)

---

### Stage 5B: Formatting LLM Call (Report Generation)

**Purpose**: Convert structured insights into polished, actionable creative strategy reports

**Input**: `llm_reports/analysis/insights.json`

**Process**:

```python
prompt = f"""
You are a creative strategy consultant. Generate polished markdown reports.

HASHTAG: {hashtag}
BUCKET: {bucket}

INPUT INSIGHTS:
{insights_json}

TASK:
Generate 5 markdown reports with professional formatting.
Each report should be clearly separated with "---REPORT: filename---" markers.

---REPORT: rf_feature_importance.md---

# Random Forest Feature Importance: {hashtag} ({bucket})

## Executive Summary
[2-3 sentences on what drives success for this bucket]

## Top 5 Success Drivers

### 1. {{feature.name}} (Importance: {{feature.importance}})

**What it means**: {{feature.interpretation}}

**The Gap**:
- Top performers: {{feature.top_performer_avg}}
- Bottom performers: {{feature.bottom_performer_avg}}
- Difference: {{feature.gap}}

**Action**: {{feature.actionable_advice}}

[Repeat for features 2-5]

## Improvement Opportunities

### {{opportunity.feature}}
**Current situation**: {{opportunity.current_gap}}
**Recommendation**: {{opportunity.recommendation}}

[Repeat for top 3 opportunities]

---REPORT: strategy_1_the_educator.md---

# Strategy 1: The Educator

## Overview
{{cluster.strategy_summary}}

**Success Metrics**:
- Videos using this pattern: {{cluster.video_count}}
- Average engagement: {{cluster.avg_engagement}}
- Top performer rate: {{cluster.top_performer_percentage * 100}}%

## Key Features

### Hook (First 3 seconds)
[List defining characteristics from cluster.defining_features.hook]

### Middle Segments
[List defining characteristics from cluster.defining_features.middle]

### Closing (Last 3 seconds)
[List defining characteristics from cluster.defining_features.closing]

## Example Videos
[List example_video_ids with brief descriptions]

## Creation Checklist
- [ ] Hook: Maintain direct eye contact (0.85+ rate)
- [ ] Hook: Use moderate pacing (3-4 scene cuts)
- [ ] Middle: High word density (50-60 words)
- [ ] Middle: Keep consistent emotional tone
- [ ] Closing: High energy delivery (0.8+ energy level)

---REPORT: strategy_2_visual_storyteller.md---
[Similar structure for cluster 1]

---REPORT: strategy_3_personal_journey.md---
[Similar structure for cluster 2]

---REPORT: bucket_summary.md---

# Bucket Summary: {hashtag} ({bucket})

## Pattern Overview

We identified 3 distinct creative strategies:

1. **{{cluster_1.name}}** ({{cluster_1.percentage}}%) - {{cluster_1.top_performer_percentage * 100}}% top performer rate
2. **{{cluster_2.name}}** ({{cluster_2.percentage}}%) - {{cluster_2.top_performer_percentage * 100}}% top performer rate
3. **{{cluster_3.name}}** ({{cluster_3.percentage}}%) - {{cluster_3.top_performer_percentage * 100}}% top performer rate

## Recommendations

**Start here**: {{recommendations.best_for_beginners}}

**Why**: {{recommendations.reasoning}}

IMPORTANT:
- Use markdown formatting (##, ###, bold, lists)
- Keep language actionable and specific
- Include exact numbers from insights
- Each report should be standalone
"""

reports = claude_api.generate(prompt, data=insights_json)

# Parse LLM response (split by ---REPORT: markers)
parsed_reports = parse_multiple_reports(reports)

save("llm_reports/formatted/rf_feature_importance.md", parsed_reports['rf'])
save("llm_reports/formatted/strategy_1_the_educator.md", parsed_reports['strategy_1'])
save("llm_reports/formatted/strategy_2_visual_storyteller.md", parsed_reports['strategy_2'])
save("llm_reports/formatted/strategy_3_personal_journey.md", parsed_reports['strategy_3'])
save("llm_reports/formatted/bucket_summary.md", parsed_reports['summary'])
```

**Outputs**:
- `llm_reports/formatted/call_2_prompt.txt`
- `llm_reports/formatted/call_2_raw_response.json`
- `llm_reports/formatted/rf_feature_importance.md`
- `llm_reports/formatted/strategy_1_the_educator.md`
- `llm_reports/formatted/strategy_2_visual_storyteller.md`
- `llm_reports/formatted/strategy_3_personal_journey.md`
- `llm_reports/formatted/bucket_summary.md`

**LLM Calls per Hashtag**: 8 total (1 per bucket × 8 buckets)

**Why Single Formatting Call**:
- Claude can generate 5 reports in one call reliably
- All reports use same `insights.json` (no need for separate contexts)
- Faster than 5 separate calls
- Consistent tone across all reports

---

### Stage 6: PDF Generation (No LLM)

**Purpose**: Convert markdown reports to professional PDFs

**Input**: `llm_reports/formatted/*.md` (5 markdown files per bucket)

**Process**:

```python
from markdown2pdf import convert_markdown_to_pdf

# Convert each markdown file to PDF
reports = [
    "rf_feature_importance",
    "strategy_1_the_educator",
    "strategy_2_visual_storyteller",
    "strategy_3_personal_journey",
    "bucket_summary"
]

for report in reports:
    md_path = f"llm_reports/formatted/{report}.md"
    pdf_path = f"reports/{report}.pdf"

    convert_markdown_to_pdf(
        md_path,
        pdf_path,
        css_template="creative_report.css"  # Custom styling
    )
```

**Outputs** (`reports/` directory):
- `rf_feature_importance.pdf`
- `strategy_1_the_educator.pdf`
- `strategy_2_visual_storyteller.pdf`
- `strategy_3_personal_journey.pdf`
- `bucket_summary.pdf`

**Report Structure** (consistent across all buckets):
- Professional formatting (headers, lists, bold text)
- Branded styling (Tumi Labs colors, fonts)
- Actionable checklists
- Specific numbers and metrics
- Example video references

**Libraries**:
- `markdown2` or `pandoc` for markdown parsing
- `weasyprint` or `pdfkit` for PDF generation
- Custom CSS template for branding

**No LLM Calls**: This stage is pure formatting (no API costs, fast execution)

---

### LLM Call Summary

**Per Bucket**:
- Analysis calls: 2 (RF + K-Means)
- Formatting calls: 1 (all 5 reports)
- **Total per bucket**: 3 calls

**Per Hashtag** (8 buckets):
- Analysis calls: 16 (2 × 8)
- Formatting calls: 8 (1 × 8)
- **Total per hashtag**: 24 calls

**Cost Estimate** (assuming $0.15 per call):
- Per bucket: $0.45
- Per hashtag: $3.60

**Duration Estimate**:
- Analysis calls: ~30 seconds each (parallel processing possible)
- Formatting calls: ~45 seconds each
- PDF generation: ~5 seconds per report
- **Total per hashtag**: ~15-20 minutes (if sequential), ~5-8 minutes (if parallel)

**Data Sent to LLM**:
- Analysis calls: ~30KB each (60KB total per bucket)
- Formatting calls: ~5-10KB each (insights.json is smaller than raw ML data)
- **Total per hashtag**: ~560KB (well under Claude's 800KB limit)

**Why Two-Call Approach**:
- Better quality through separation of concerns (analysis vs formatting)
- Easier iteration (change formatting without re-analysis)
- Intermediate insights valuable for debugging and reuse
- Only moderate cost increase vs single-call approach


---

## CLI Configuration

All RumiAI analyses use **multiple orthogonal configuration dimensions**. These flags work independently and can be combined in any valid way.

**Command Structure**:
```bash
python rumiai_ml_batch.py \
  --client "client_name" \
  --analysis-type {hashtag|competitor|creator} \    # Target Type
  --target "{target}" \
  --analysis-mode {top|recent} \                     # Analysis Mode
  --selection-strategy {contrastive|top} \           # Selection Strategy
  --video-count N \                                  # Video Count
  --date-filter last_N_days \                        # Date Filter
  --report-type {single|comparison}                  # Report Type
```

**Design Principles**:
- **Orthogonal**: Each dimension is independent
- **Composable**: Valid combinations work across all target types
- **Default-aware**: Each target type has sensible defaults
- **Explicit**: All parameters exposed as CLI flags (automation-friendly)

---

### Target Types

Determines **what source** to analyze videos from (different Apify scrapers, different use cases).

**Available Types**:

| Type | CLI Flag | Target Format | Data Source | Primary Use Case |
|------|----------|---------------|-------------|------------------|
| `hashtag` | `--analysis-type hashtag` | `#nutrition` | TikTok hashtag search | Market research - identify viral patterns |
| `competitor` | `--analysis-type competitor` | `@rival_brand` | TikTok profile | Competitive intelligence - understand rivals |
| `creator` | `--analysis-type creator` | `@potential_affiliate` | TikTok profile | Creator vetting - assess fit for hiring |

**CLI Usage**:
```bash
--analysis-type hashtag     # Analyze hashtag content
--analysis-type competitor  # Analyze competitor profile
--analysis-type creator     # Analyze creator profile
```

**Key Differences**:

| Aspect | Hashtag | Competitor | Creator |
|--------|---------|------------|---------|
| **Apify Scraper** | clockworks/tiktok-hashtag-scraper | clockworks/tiktok-scraper | clockworks/tiktok-scraper |
| **Video Source** | All TikTok users posting with hashtag | Single profile's content | Single profile's content |
| **ML Training** | Yes (classification models) | Optional (descriptive only) | No (uses existing models) |
| **Default Mode** | `top` | `top` | `recent` |
| **Default Strategy** | `contrastive` | `top` | `top` |
| **Default Date Filter** | `last_90_days` | `last_90_days` | `last_30_days` |
| **Default Video Count** | 100 | 40 | 40 |

**Why Target Type Matters**:
- Different scrapers (hashtag scraper vs profile scraper)
- Different business questions (market trends vs competitor benchmarking vs hiring decisions)
- Different default configurations (optimized per use case)
- Same underlying processing pipeline (RumiAI → Buckets → ML → Reports)

---

### Analysis Modes

RumiAI supports multiple analysis modes to answer different business questions. The mode controls how Apify sorts and selects videos.

**Available Modes**:

| Mode | Sort By | Use Case | Default For |
|------|---------|----------|-------------|
| `top` | Engagement (composite score) | "What works?" - Identify successful patterns | Hashtag, Competitor |
| `recent` | Publish date (newest first) | "What's happening now?" - Track current trends | Creator |

**CLI Usage**:
```bash
--analysis-mode top     # Analyze highest-performing content
--analysis-mode recent  # Analyze most recent content
```

**Detailed Information**: See [SelectionStrategies.md.md](./documentation_migration/FutureDevelopments/SelectionStrategies.md.md)

---

### Selection Strategies

Determines **what videos to select** after sorting (orthogonal to analysis mode).

**Available Strategies**:

| Strategy | Videos Selected | Use Case | Default For |
|----------|----------------|----------|-------------|
| `contrastive` | Top 80% + Bottom 20% per bucket (N configurable, default 100) | ML training - identify pattern differences through contrast | Hashtag |
| `top` | Top N per bucket only (N configurable, default 40) | Best practices analysis - learn from success only | Competitor, Creator |

**CLI Parameter**: `--video-count N`
- Controls how many videos to analyze per winning bucket
- Contrastive: N split as 80% top + 20% bottom (e.g., N=100 → 80+20)
- Top: N all top performers (e.g., N=40 → top 40)
- **Success-based bucket selection**: Analyzes top 100 performers to identify where winners cluster
- Only top 3 buckets where winners concentrate are processed (not volume-based)

**Why Separate from Analysis Mode?**
These are orthogonal dimensions:
- **Analysis Mode** (top/recent): Controls HOW videos are sorted
- **Selection Strategy** (contrastive/top): Controls WHAT subset is analyzed

**Detailed Strategy Design**: See [SelectionStrategies.md](./SelectionStrategies.md) for comprehensive strategy documentation, adaptive processing logic, and business trade-offs.

---

### Date Filtering

Controls **when videos were published** - filters scraped videos by publication date (orthogonal to analysis mode and selection strategy).

**CLI Parameter**: `--date-filter last_N_days`

**Format**: Relative date range only
- `last_N_days` where N is the number of days to look back from today
- Examples: `last_30_days`, `last_90_days`, `last_180_days`

**Default**: `last_90_days`

**CLI Usage**:
```bash
--date-filter last_90_days   # Last 90 days (default)
--date-filter last_30_days   # Last 30 days
--date-filter last_180_days  # Last 180 days
```

**How It Works**:
1. Apify scrapes 800 videos from target (no server-side date filtering available)
2. **Client-side filtering**: System filters videos by `create_time` based on date filter
3. Filtered videos proceed to bucketing and selection

**Why Client-Side?**
- Apify's hashtag scraper doesn't support server-side date filtering
- Profile scraper has date support, but client-side used for consistency across all target types
- Ensures uniform behavior regardless of scraper type

**Business Value**:
- **Recency control**: Focus on recent trends vs historical patterns
- **Seasonal analysis**: Analyze specific time periods (e.g., holiday season)
- **Trend detection**: Track how patterns evolve over time
- **Data quality**: Exclude outdated content that may skew insights

**Interaction with Analysis Modes**:
- **Top Mode + Date Filter**: "What worked recently?" (best practices from last N days)
- **Recent Mode + Date Filter**: "What's happening now?" (most recent content within last N days)
- Both dimensions are orthogonal - date filters WHEN, mode filters HOW

**Default Per Target Type**:
- **Hashtag**: `last_90_days` (quarterly trends for market research)
- **Competitor**: `last_90_days` (current competitive strategies)
- **Creator**: `last_30_days` (recent natural style for vetting)

**Example Impact**:
```
Scraped: 800 videos (all-time)
↓ Apply date_filter: last_90_days
Filtered: 600 videos (within date range)
↓ Analyze top 100 performers (success-based distribution)
Top 100 winners: 18-33s (45%), 33-60s (30%), 13-18s (20%), 9-13s (5%)
↓ Select top 3 winning buckets
Process: 18-33s, 33-60s, 13-18s (95% of winners)
↓ Apply selection strategy (contrastive, N=100)
Per bucket: 100 videos (80 top + 20 bottom)
```

**Technical Implementation**: See [MLAnalysisModeTI.md - Date Filter Implementation](./MLAnalysisModeTI.md)

---

### Report Types

Determines **what type of output** is generated (orthogonal to target type, analysis mode, and selection strategy).

**Available Types**:

| Type | What It Does | Prerequisites | Output | Applies To |
|------|-------------|---------------|--------|------------|
| `single` | Deep analysis of one target | None | Full ML analysis + creative reports | All target types (default) |
| `comparison` | Side-by-side comparison of 2+ targets | All targets must have existing single analyses | LLM-synthesized comparison report | All target types |

**CLI Usage**:
```bash
--report-type single       # Deep dive on one target (default)
--report-type comparison   # Compare multiple targets
```

**Single Mode Process**:
1. Scrape videos from target (Apify)
2. Run full ML pipeline (RumiAI → Buckets → ML Training)
3. Generate creative reports per bucket
4. Output: Models + Reports + Analysis data

**Comparison Mode Process**:
1. **No video processing** - uses existing analyses
2. Load data from completed single analyses
3. Send to Claude API with comparison prompt
4. Generate comparison report
5. Output: Single comparison PDF

**Key Differences**:

| Aspect | Single | Comparison |
|--------|--------|------------|
| **Video Processing** | Full ML pipeline (hours) | No processing (seconds) |
| **Prerequisites** | None | Requires existing single analyses |
| **LLM Calls** | 24 per target (analysis + formatting) | 1 per comparison group |
| **Output** | ML models + 40 creative reports | 1 comparison PDF |
| **Duration** | 6-8 hours (hashtag, 300 videos) | ~30 seconds (LLM only) |
| **Cost** | Apify ($4) + Compute + LLM ($3.60) | LLM only (~$0.50) |

**Error Handling (Comparison Mode)**:
```bash
# If any target hasn't been analyzed individually:
✗ Cannot generate comparison report

Missing individual analyses:
  ✓ #nutrition - analyzed (2025-01-28)
  ✓ #fitness - analyzed (2025-01-27)
  ✗ #wellness - NOT ANALYZED

Run individual analysis first:
  python rumiai_ml_batch.py --analysis-type hashtag --target "#wellness" --date-filter last_90_days --report-type single
```

**Checkpoint/Resume**: Auto-resume on restart (detects checkpoint, continues automatically). Use `--force` flag to discard checkpoint and restart fresh. See [MLCheckpointResume.md](./documentation_migration/FutureDevelopments/MLCheckpointResume.md) for details.


---

# New Features

## 1. Creator Match Analysis

**Purpose**: Analyze potential affiliate creators' natural content style and match against client's viral patterns to optimize hiring decisions.

**Key Capabilities**:
- Analyze most recent 40 videos from creator's TikTok handle
- Identify creator's natural duration distribution across 8 buckets
- Compare creator's production style against client hashtag/competitor success patterns
- Generate compatibility scores combining distribution match + feature alignment
- Provide hiring recommendation tiers (Immediate Hire → Pass)

**Business Value**:
- Reduce hiring risk by identifying natural creator-brand fit
- Minimize coaching overhead by selecting creators who naturally produce winning durations
- Data-driven affiliate vetting instead of gut-feel decisions

**Implementation Details**: See [MLCreatorMatch.md](./documentation_migration/FutureDevelopments/MLCreatorMatch.md)

**Priority**: HIGH - Critical for affiliate vetting and ROI optimization

---
## 2. Checkpoint Resume System

**Purpose**: Enable recovery from process interruptions during long-running batch analyses (6-8 hours) without re-processing completed videos.

**Key Capabilities**:
- Automatic checkpoint saving after each video completes analysis
- Auto-resume detection when restarting interrupted batch processing
- Track completed videos, failed videos, and current processing state per bucket
- Support for manual restart via `--force` flag (discard checkpoint)

**Business Value**:
- Prevent wasted compute time (3-6 hours saved per interruption)
- Enable reliable batch processing despite SSH disconnects, crashes, or manual stops
- Reduce risk for long-running client analyses (300+ videos)

**Implementation Details**: See [MLCheckpointResume.md](./documentation_migration/FutureDevelopments/MLCheckpointResume.md)

**Priority**: HIGH - Critical for 6-8 hour batch jobs reliability

---
## 3. LLM Data Strategy

**Purpose**: Define optimal data formatting and aggregation strategy for sending ML analysis results (Random Forest + K-Means) to Claude API for insight generation and report creation.

**Key Capabilities**:
- Support for two ML analysis types per bucket (Random Forest feature importance + K-Means clustering)
- Full raw data format: Send complete video-level features (N videos × 35 features per JSON, N from --video-count)
- Aggregated statistics format: Send compressed statistical summaries (mean, median, quartiles, distribution)
- Token limit management: Stay within Claude API's 200K token (~800KB) limit
- Scalable comparison mode: Support 5-10+ hashtag comparisons via aggregation

**Architecture Decisions**:

**Single Hashtag Analysis**:
- Data volume: ~480KB (2 JSONs per bucket × 8 buckets)
- Recommendation: **Full raw data**
- Rationale: Well within token limits, provides richer insights for LLM
- LLM calls: 16 total (RF + K-Means per bucket)

**Multi-Hashtag Comparison**:
- Data volume: ~1.44MB raw (exceeds limits)
- Recommendation: **Aggregated statistics** (reduces to ~200KB)
- Rationale: Token limit compliance, prevents hallucination risk
- LLM calls: 8 total (combined RF + K-Means per bucket)

**Business Value**:
- Prevent hallucination and poor report quality from oversized context windows
- Enable scalable multi-hashtag comparisons (5-10+ hashtags)
- Balance data richness (full raw) vs efficiency (aggregated) based on analysis type
- Optimize LLM API costs by right-sizing payloads

**Implementation Details**:
- High-Level Design: [ML_LLMData.md](./documentation_migration/FutureDevelopments/ML_LLMData.md)
- Technical Implementation: [ML_LLMDataTI.md](./documentation_migration/FutureDevelopments/ML_LLMDataTI.md)

**Priority**: MEDIUM - Required before report generation, but can start with single hashtag MVP (full raw data approach)

**Dependencies**:
- Requires ML analysis outputs (Random Forest + K-Means JSONs per bucket)
- Feeds into Creative Report Generation system
- Integration with Claude API for insight synthesis

---

