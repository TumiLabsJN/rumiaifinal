# RumiAI Production Pipeline Flow

**Purpose**: Authoritative map of actual production code flow for LLM agents revising the orchestrator
**Source**: Generated from systematic code analysis of all pipeline stages
**Last Updated**: 2025-01-28

---

## Quick Navigation

- [Pipeline Overview](#pipeline-overview)
- [Stage Dependencies Graph](#stage-dependencies-graph)
- [Critical Path Analysis](#critical-path-analysis)
- [Stage Contracts](#stage-contracts)
- [File Lifecycle Map](#file-lifecycle-map)
- [Checkpoint Strategy](#checkpoint-strategy)
- [Error Propagation Matrix](#error-propagation-matrix)

---

## Pipeline Overview

### Execution Sequence

```
Stage 1 (Video Discovery)
    ↓
Stage 2 (ML Processing) - 9 ML Services
    ↓
Stage 2.5 (File Organization)
    ↓
Stage 2.5.1 (Transcript Validation)
    ↓
Stage 2.6 (Content Discovery)
    ↓ **MANUAL CURATION (~15 min)** ← BLOCKS PIPELINE
Stage 2.7 (Content Classification)
    ↓
Stage 3 (Feature Aggregation)
    ↓
Stage 3.4 (Review CSV Generation)
    ↓
Stage 4 (Feature Transformation)
    ↓
Stage 5 (Model Training)
    ↓
Stage 6 (ML Analysis Generation)
    ↓
Stage 7 (LLM Analysis)
    ↓
Stage 8 (Report Generation) ← PLANNED, NOT IMPLEMENTED
```

### Stage Count by Type

- **Data Collection**: Stage 1
- **ML Feature Extraction**: Stage 2 (9 services)
- **Data Organization**: Stages 2.5, 2.5.1
- **Content Analysis**: Stages 2.6, 2.7
- **ML Training Pipeline**: Stages 3, 3.4, 4, 5, 6
- **Analysis & Reporting**: Stages 7, 8

### Total Processing Time

- **Full Pipeline**: ~10-15 minutes for 120 videos (depending on parallel mode)
- **Bottleneck**: Stage 2 FEAT emotion detection (43% of ML time)
- **Manual Intervention**: Stage 2.6 → 2.7 (~15 min for taxonomy curation)

---

## Stage Dependencies Graph

### Visual Dependency Map

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Video Discovery                                    │
│ Output: winner_analysis.json, selection_manifest.json       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: ML Processing (9 Services)                         │
│ Output: temporal_windows_updated.json (flat dir)            │
│ ⚠️ HARDCODED: /home/jorge/rumiaifinal/insights/             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2.5: File Organization                                │
│ Output: Moves files to buckets/, creates manifest           │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2.5.1: Transcript Validation                          │
│ Output: Filters invalid transcripts, updates manifest       │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2.6: Content Discovery                                │
│ Output: {hashtag}_raw_discovery.json                        │
└───────────────┬─────────────────────────────────────────────┘
                ↓ **MANUAL CURATION REQUIRED**
┌─────────────────────────────────────────────────────────────┐
│ HUMAN: Edit {hashtag}_taxonomy.json                         │
│ Time: ~15 minutes                                            │
│ Pipeline Status: EXIT CODE 2 (Paused)                       │
└───────────────┬─────────────────────────────────────────────┘
                ↓ **RESUME PIPELINE**
┌─────────────────────────────────────────────────────────────┐
│ Stage 2.7: Content Classification                           │
│ Output: validated/bucket_{name}/*_content.json (120 files)  │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Feature Aggregation                                │
│ Output: aggregated_features.csv (350+ features)             │
└────────────┬──────────────────────────────────────┬─────────┘
             ↓                                      ↓
┌────────────────────────────┐   ┌──────────────────────────┐
│ Stage 3.4: Review CSV      │   │ Stage 4: Transformation  │
│ Output: video_review.csv   │   │ Output: 13 files/bucket  │
└────────────────────────────┘   └──────────┬───────────────┘
                                             ↓
                              ┌──────────────────────────────┐
                              │ Stage 5: Model Training      │
                              │ Output: .pkl models          │
                              └──────────┬───────────────────┘
                                         ↓
                              ┌──────────────────────────────┐
                              │ Stage 6: ML Analysis         │
                              │ Output: analysis JSONs       │
                              └──────────┬───────────────────┘
                                         ↓
                              ┌──────────────────────────────┐
                              │ Stage 7: LLM Analysis        │
                              │ Output: winning_formulas.json│
                              └──────────┬───────────────────┘
                                         ↓
                              ┌──────────────────────────────┐
                              │ Stage 8: Report Generation   │
                              │ Status: 📝 PLANNED           │
                              └──────────────────────────────┘
```

---

## Critical Path Analysis

### Blocking Dependencies

| Stage | Blocks | Reason | Workaround |
|-------|--------|--------|------------|
| **Stage 1** | All downstream | No videos = no processing | Must complete first |
| **Stage 2.6** | Stage 2.7+ | Manual taxonomy curation required | Pipeline exits with code 2 |
| **Stage 2.7** | Stage 7, 8 | Content classifications needed for reports | Cannot skip |
| **Stage 3** | Stages 4-7 | aggregated_features.csv is ML input | Must complete |

### Parallel Processing Opportunities

- **Stage 2**: 9 ML services run **sequentially by default**, parallel mode via `ENABLE_PARALLEL_CLASSIFICATION`
- **Stage 2.7**: Classification can run parallel (env: `MAX_CLASSIFICATION_WORKERS`)
- **Stage 7 Phase 1**: Window analyses run sequentially (max_workers=1) to avoid API rate limits
- **Per-Bucket Processing**: Each stage processes multiple winning buckets in sequence (skip-on-fail)

### Critical Timing Thresholds

- **Stage 2**: ~60-80s per 60s video (FEAT = 43% of time)
- **Stage 2.7**: ~5 min sequential, ~2 min parallel (120 videos)
- **Stage 4**: <30s per bucket (baseline)
- **Stage 7**: ~90s per window (Anthropic API timeout)

---

## Stage Contracts

### Stage 1: Video Discovery

**Implementation**: [`ml_pipeline/stage1_discovery/video_discovery.py`](ml_pipeline/stage1_discovery/video_discovery.py)
**Entry Point**: `VideoDiscovery.run()` (line 83-312)
**Orchestrator Call**: [`rumiai_ml_batch.py:570-707`](rumiai_ml_batch.py#L570-L707)

**Inputs**:
- CLI args: `--client`, `--target`, `--video-count`, `--strategy`
- Environment: `APIFY_API_KEY`

**Outputs**:
```
{analysis_base}/
├── winner_analysis.json              # Top 3 winning buckets by video count
├── selection_manifest.json           # 120 video IDs split by bucket+performer
└── buckets/
    ├── bucket_18-33s/
    │   └── selected_videos.json      # TikTok API metadata (40 videos)
    ├── bucket_33-60s/
    │   └── selected_videos.json
    └── bucket_60-90s/
        └── selected_videos.json
```

**Key Functions**:
- `VideoDiscovery.run()` - Main entry point
- `identify_winning_buckets()` - Select top 3 by count
- `create_selection_manifest()` - Split 80/20 top/bottom

**Checkpoint**: `{analysis_base}/checkpoints/stage_1_checkpoint.json`

**Depends On**: None (first stage)

**Consumed By**:
- Stage 2 (video files)
- Stage 2.5 (manifest)
- Stage 2.6, 2.7 (manifest)
- Stage 8 (metadata)

**Error Strategy**: Exit pipeline on failure (exit code 1)

**Skip Logic**: Validates checkpoint schema (`winning_buckets`, `output_files`, `timestamp`) + file existence

---

### Stage 2: Video Processing (ML Services)

**Implementation**: [`ml_pipeline/stage2_processing/main.py`](ml_pipeline/stage2_processing/main.py)
**Entry Point**: `stage_2_video_processing_main()` (line 91-184)
**Orchestrator Call**: [`rumiai_ml_batch.py:708-776`](rumiai_ml_batch.py#L708-L776)

**Inputs**:
- Stage 1: `buckets/bucket_{name}/selected_videos.json`
- Video files: Downloaded by Stage 2 to `{bucket}/videos/`

**Outputs** (⚠️ HARDCODED PATH):
```
/home/jorge/rumiaifinal/insights/
└── {video_id}_temporal_windows_updated.json  # 9 ML services aggregated
```

**ML Services** (9 total):
1. YOLO Object Detection
2. Whisper Speech Transcription
3. MediaPipe Pose/Gesture
4. OCR Text Detection
5. Scene Detection
6. Audio Energy Analysis
7. FEAT Emotion Detection (43% of total time)
8. DeepFace Gender Classification
9. MediaPipe Face Landmarks

**Key Files**:
- `video_analyzer.py` - Service orchestration
- `timeline_builder.py` - Timeline unification
- `temporal_compute.py` - Window feature extraction

**Checkpoint**: `{bucket_path}/checkpoints/stage_2_checkpoint.json`

**Depends On**: Stage 1 (video metadata)

**Consumed By**: Stage 2.5 (moves files to buckets)

**Error Strategy**: Skip-on-fail per bucket, continue with remaining

**Critical Path**: FEAT emotion detection is bottleneck (~40s per 60s video)

---

### Stage 2.5: File Organization

**Implementation**: [`ml_pipeline/stage2_5_organize/file_organizer.py`](ml_pipeline/stage2_5_organize/file_organizer.py)
**Entry Point**: `stage_2_5_file_organization_main()` (line 91-452)
**Orchestrator Call**: [`rumiai_ml_batch.py:777-806`](rumiai_ml_batch.py#L777-L806)

**Inputs**:
- Stage 1: `winner_analysis.json`
- Stage 2: `/home/jorge/rumiaifinal/insights/*_temporal_windows_updated.json`

**Outputs**:
```
{analysis_base}/
├── selection_manifest.json           # ⚠️ CRITICAL: Used by 2.6, 2.7, Stage 8
└── buckets/
    └── bucket_{name}/
        └── analysis/
            └── insights/
                └── {video_id}_temporal_windows_updated.json  # Moved from flat dir
```

**Key Functions**:
- `file_organizer.py::organize_files()` - Main entry
- `create_selection_manifest()` - Critical for downstream stages

**Checkpoint**: None (tracked via moved file counts)

**Depends On**: Stage 1 (bucket list), Stage 2 (temporal_windows files)

**Consumed By**: Stage 2.5.1, 2.6, 2.7, 3

**Error Strategy**: Skip-on-fail (logs missing files, continues)

---

### Stage 2.5.1: Transcript Validation

**Implementation**: [`ml_pipeline/stage2_content_analysis/validation.py`](ml_pipeline/stage2_content_analysis/validation.py)
**Orchestrator Call**: [`rumiai_ml_batch.py:807-860`](rumiai_ml_batch.py#L807-L860)

**Inputs**:
- Stage 2: Whisper transcripts from `unified_analysis/{video_id}.json`

**Outputs**:
- Updated `selection_manifest.json` (removes invalid videos)
- Validation summary (invalid count, reasons breakdown)

**Validation Criteria**:
- Min 10 words spoken
- Not music/noise-only (Whisper confidence check)
- Valid speech detected (not [MUSIC], [NOISE] tags)

**Minimum Threshold**: <30 valid videos → Pipeline fails (exit code 1)

**Depends On**: Stage 2 (Whisper transcripts), Stage 2.5 (manifest)

**Consumed By**: Stage 2.6, 2.7 (filtered manifest)

**Error Strategy**: Exit on threshold failure

---

### Stage 2.6: Content Discovery

**Implementation**: [`ml_pipeline/stage2_content_analysis/discovery.py`](ml_pipeline/stage2_content_analysis/discovery.py)
**Entry Point**: `run_discovery_stage()` (line 51-642)
**Orchestrator Call**: [`rumiai_ml_batch.py:862-936`](rumiai_ml_batch.py#L862-L936)

**Inputs**:
- Stage 2.5.1: `selection_manifest.json` (validated)
- Stage 2: Whisper transcripts, unified_analysis captions

**Outputs**:
```
{analysis_base}/content_taxonomies/
├── {hashtag}_raw_discovery.json      # LLM-generated taxonomy (7 categories)
└── {hashtag}_taxonomy.json           # 🔴 MANUAL CURATION REQUIRED
```

**⚠️ CRITICAL BLOCKING POINT**:
- Discovery runs **ONE TIME ONLY** (checks `.content_analysis_state.json`)
- Pipeline **EXITS WITH CODE 2** after creating raw taxonomy
- User must manually curate taxonomy (~15 min)
- Pipeline resumes from Stage 2.7 after curation

**Taxonomy Categories** (7 required fields):
1. `content_categories` - Video topic types
2. `hook_strategies` - First 3 seconds patterns
3. `caption_structures` - Caption formats
4. `visual_patterns` - Visual composition styles
5. `audio_patterns` - Sound/music usage
6. `keywords` - Common hashtags/terms
7. `content_tactics` - Engagement techniques

**State Tracking**: `.content_analysis_state.json`
```json
{
  "discovery_complete": true,
  "taxonomy_curated": false,  // User sets to true after manual edit
  "taxonomy_version": "1.0"
}
```

**Depends On**: Stage 2.5.1 (validated manifest), Stage 2 (transcripts)

**Consumed By**: Stage 2.7 (taxonomy for classification)

**Error Strategy**: One-time execution, blocks pipeline until manual curation

---

### Stage 2.7: Content Classification

**Implementation**: [`ml_pipeline/stage2_content_analysis/classification.py`](ml_pipeline/stage2_content_analysis/classification.py)
**Entry Point**: `classify_videos()` (line 102-1662)
**Orchestrator Call**: [`rumiai_ml_batch.py:937-1016`](rumiai_ml_batch.py#L937-L1016)

**Inputs**:
- Stage 2.6: `{hashtag}_taxonomy.json` (manually curated)
- Stage 2.5.1: `selection_manifest.json`
- Stage 2: Whisper transcripts, unified_analysis captions

**Outputs**:
```
{analysis_base}/content_analysis/validated/
├── bucket_18-33s/
│   ├── {video_id}_content.json       # 15 fields per video
│   └── ... (40 files)
├── bucket_33-60s/
│   └── ... (40 files)
└── bucket_60-90s/
    └── ... (40 files)
```

**Output Schema** (15 fields per video):
```json
{
  "video_id": "7545713916584774968",
  "bucket": "18-33s",
  "performer_type": "top",           // ⚠️ CRITICAL: For Stage 8 filtering
  "content_category": "Recipe Tutorial",
  "hook_strategy": "Problem-Solution",
  "closing_strategy": "Call to Action",
  "visual_pattern": "Close-up Hands",
  "audio_pattern": "Voiceover",
  "caption_analysis": {
    "hook_type": "question",
    "caption_length": "short",
    "emoji_usage": "some",
    "hashtag_count": 7
  },
  "caption_cta_type": "link_in_bio",
  "keywords": ["#guthealth", "#protein"],
  "content_tactics": ["Personal testimony"],
  // ... 15 total fields
}
```

**Processing Mode**:
- Sequential: ~5 min for 120 videos
- Parallel: ~2 min (env: `ENABLE_PARALLEL_CLASSIFICATION=true`, `MAX_CLASSIFICATION_WORKERS=4`)

**Checkpoint**: `.checkpoints/classification_checkpoint.json` (thread-safe)

**Depends On**: Stage 2.6 (taxonomy), Stage 2.5.1 (manifest), Stage 2 (transcripts)

**Consumed By**: Stage 7 (content insights), Stage 8 (all reports)

**Error Strategy**: Skip-on-fail per video, checkpoint/resume enabled

---

### Stage 3: Feature Aggregation

**Implementation**: [`scripts/stage3_aggregation.py`](scripts/stage3_aggregation.py)
**Entry Point**: `aggregate_features()` (line 45-931)
**Orchestrator Call**: [`rumiai_ml_batch.py:1018-1194`](rumiai_ml_batch.py#L1018-L1194)

**Inputs**:
- Stage 2.5: `buckets/bucket_{name}/analysis/insights/*_temporal_windows_updated.json`

**Outputs**:
```
{bucket_path}/ml_analysis/
├── aggregated_features.csv           # 350+ features × 40 videos
└── aggregation_summary.json          # Metadata
```

**Feature Count by Bucket** (temporal features only):
- 0-3s bucket: 21 features × 1 window = 21 + 3 metadata = 24 columns
- 3-9s bucket: 21 × 2 + 3 = 45 columns
- 18-33s bucket: 21 × 6 + 3 + 5 cross-window + 1 label = **135 columns**

**Key Functions**:
- `extract_window_features()` - Parse temporal_windows (line 120)
- `aggregate_features()` - Main CSV creation

**Checkpoint**: `{bucket_path}/checkpoints/stage_3_checkpoint.json`

**Depends On**: Stage 2.5 (organized temporal_windows files)

**Consumed By**: Stage 3.4 (review CSV), Stage 4 (transformation)

**Error Strategy**: Skip bucket on ValueError/AssertionError, exit on IOError

---

### Stage 3.4: Review CSV Generation

**Implementation**: [`ml_pipeline/stage3_aggregation/review_csv_generator.py`](ml_pipeline/stage3_aggregation/review_csv_generator.py)
**Entry Point**: `generate_review_csv_for_bucket()` (line 30)
**Orchestrator Call**: [`rumiai_ml_batch.py:1120-1145`](rumiai_ml_batch.py#L1120-L1145)

**Inputs**:
- Stage 3: `aggregated_features.csv`
- Stage 1: `selected_videos.json` (TikTok URLs)

**Outputs**:
```
{bucket_path}/validation/
└── video_review.csv                  # Video ID + TikTok URL for manual outlier review
```

**Purpose**: Manual outlier inspection (optional human QA step)

**Depends On**: Stage 3 (aggregated CSV)

**Consumed By**: Human review (optional)

**Error Strategy**: Best-effort (logged warnings, doesn't block Stage 4)

---

### Stage 4: Feature Transformation

**Implementation**: [`rumiai_v2/processors/feature_transformation.py`](rumiai_v2/processors/feature_transformation.py)
**Entry Point**: `run_stage4_transformation()` (line 874-1088)
**Orchestrator Call**: [`rumiai_ml_batch.py:1195-1386`](rumiai_ml_batch.py#L1195-L1386)

**Inputs**:
- Stage 3: `ml_analysis/aggregated_features.csv`

**Outputs** (13 files per bucket):
```
{bucket_path}/ml_analysis/
├── rf_transformed.csv                # Video-level RF (147 features for 18-33s)
├── hook_rf_transformed.csv           # Window-level RF (22 features)
├── middle_1_rf_transformed.csv
├── middle_2_rf_transformed.csv
├── middle_3_rf_transformed.csv
├── closing_rf_transformed.csv
├── hook_km_transformed.csv           # Window-level K-Means (27 features)
├── middle_1_km_transformed.csv
├── middle_2_km_transformed.csv
├── middle_3_km_transformed.csv
├── closing_km_transformed.csv
├── hook_scalers.pkl                  # MinMaxScaler objects
├── middle_1_scalers.pkl
├── middle_2_scalers.pkl
├── middle_3_scalers.pkl
└── closing_scalers.pkl
```

**Transformations**:
1. **Video-Level RF**: One-hot emotions, temporal features, gender encoding, cross-window features
2. **Window-Level RF**: Extract single window, add `is_top_performer` label
3. **Window-Level K-Means**: Log1p+MinMax scale, normalize to [0-1]

**Key Functions**:
- `validate_input()` - Pre-flight checks (line 239)
- `transform_video_level_rf()` - Video-level transformation (line 433)
- `transform_window_level_rf()` - Window RF (line 545)
- `transform_window_level_kmeans()` - Window K-Means (line 603)
- `validate_outputs_and_checkpoint()` - Post-validation (line 774)

**Checkpoint**: `{bucket_path}/checkpoints/stage_4_checkpoint.json`

**Depends On**: Stage 3 (aggregated CSV)

**Consumed By**: Stage 5 (training)

**Error Strategy**: Skip bucket on ValueError/AssertionError, exit on IOError/TimeoutError

**Fallback Logic**: If checkpoint missing but aggregated CSV exists, validates CSV and proceeds

---

### Stage 5: Model Training

**Implementation**: [`rumiai_v2/processors/model_training.py`](rumiai_v2/processors/model_training.py)
**Entry Point**: `run_stage5_training()` (line 970-1062)
**Orchestrator Call**: [`rumiai_ml_batch.py:1387-1560`](rumiai_ml_batch.py#L1387-L1560)

**Inputs**:
- Stage 4: All 13 transformation CSVs + scalers
- Config: `config/model_hyperparameters.json` (graceful fallback to defaults)

**Outputs**:
```
{bucket_path}/models/
├── rf_video_{bucket}.pkl             # Video-level Random Forest
├── rf_hook_{bucket}.pkl              # Window-level RF (6 models)
├── rf_middle_1_{bucket}.pkl
├── rf_middle_2_{bucket}.pkl
├── rf_middle_3_{bucket}.pkl
├── rf_closing_{bucket}.pkl
├── hook_kmeans_{bucket}.pkl          # Window-level K-Means (6 models)
├── middle_1_kmeans_{bucket}.pkl
├── middle_2_kmeans_{bucket}.pkl
├── middle_3_kmeans_{bucket}.pkl
├── closing_kmeans_{bucket}.pkl
├── hook_X_data_{bucket}.pkl          # Saved feature matrices (6 files)
├── middle_1_X_data_{bucket}.pkl
├── middle_2_X_data_{bucket}.pkl
├── middle_3_X_data_{bucket}.pkl
├── closing_X_data_{bucket}.pkl
├── hook_scalers_{bucket}.pkl         # Scalers (6 files, copied from Stage 4)
├── middle_1_scalers_{bucket}.pkl
├── middle_2_scalers_{bucket}.pkl
├── middle_3_scalers_{bucket}.pkl
├── closing_scalers_{bucket}.pkl
└── model_metrics.json                # Performance metrics
```

**Model Counts** (per bucket):
- **Contrastive mode**: 1 video RF + 6 window RF + 6 K-Means = **13 models**
- **Top mode**: 6 K-Means only (RF skipped - single class)

**Hyperparameters** (default):
```json
{
  "random_forest": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "kmeans": {
    "n_clusters": 3,
    "random_state": 42,
    "n_init": 10
  }
}
```

**Key Functions**:
- `validate_stage4_outputs()` - Pre-flight (line 84)
- `load_model_config()` - Hyperparameters (line 150)
- `train_bucket_models()` - Main training loop
- `generate_model_metrics()` - Performance tracking (line 284)

**Checkpoint**: `{bucket_path}/checkpoints/stage_5_checkpoint.json`

**Depends On**: Stage 4 (transformed CSVs)

**Consumed By**: Stage 6 (model analysis)

**Error Strategy**:
- **Custom exceptions** (StageInputError, InsufficientDataError, ModelTrainingError) → Skip bucket
- **IOError** → Exit pipeline

**Atomic Rollback**: On failure, all models deleted (all-or-nothing per bucket)

---

### Stage 6: ML Analysis Generation

**Implementation**: [`ml_pipeline/stage6_analysis/ml_analysis_generation.py`](ml_pipeline/stage6_analysis/ml_analysis_generation.py)
**Entry Point**: `generate_ml_analysis_jsons()` (line 695-795)
**Orchestrator Call**: [`rumiai_ml_batch.py:1562-1750`](rumiai_ml_batch.py#L1562-L1750)

**Inputs**:
- Stage 5: All trained models (13 .pkl files)
- Stage 4: Transformed CSVs
- Stage 3: `aggregated_features.csv`

**Outputs** (per bucket):
```
{bucket_path}/ml_analysis/
├── rf_video_analysis.json            # Video-level feature importance (top 10)
├── hook_rf_analysis.json             # Window-level RF (top 10 per window)
├── middle_1_rf_analysis.json
├── middle_2_rf_analysis.json
├── middle_3_rf_analysis.json
├── closing_rf_analysis.json
├── hook_kmeans_analysis.json         # Window-level K-Means (3 clusters per window)
├── middle_1_kmeans_analysis.json
├── middle_2_kmeans_analysis.json
├── middle_3_kmeans_analysis.json
└── closing_kmeans_analysis.json
```

**Output Counts**: 1 video RF + 6 window RF + 6 K-Means = **13 JSON files per bucket**

**JSON Schemas**:

**RF Analysis** (`{window}_rf_analysis.json`):
```json
{
  "analysis_type": "random_forest",
  "bucket": "18-33s",
  "window": "hook",
  "video_count": 40,
  "input_features": 21,
  "feature_importance": [
    {
      "feature": "eye_contact_rate",
      "importance": 0.12,
      "top_performer_avg": 0.65,
      "bottom_performer_avg": 0.42,
      "gap": 0.23,
      "distribution": {
        "thresholds": {"high": 0.75, "low": 0.50},
        "top_performers": {"high_percentage": 0.66, "medium_percentage": 0.25, "low_percentage": 0.09},
        "bottom_performers": {"high_percentage": 0.20, "medium_percentage": 0.40, "low_percentage": 0.40}
      }
    }
    // ... top 10 features
  ]
}
```

**K-Means Analysis** (`{window}_kmeans_analysis.json`):
```json
{
  "analysis_type": "kmeans",
  "bucket": "18-33s",
  "window": "hook",
  "video_count": 40,
  "cluster_count": 3,
  "silhouette_score": 0.45,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 15,
      "top_features": ["eye_contact_rate", "scene_count", "energy_level"],
      "centroid": [0.65, 0.42, 0.78, ...],  // 27 values
      "videos": [
        {"video_id": "7545713916584774968", "distance": 0.12}
        // ... all videos in cluster
      ]
    }
    // ... 3 clusters
  ]
}
```

**Pre-Flight Validation** (40+ files checked):
- Stage 4: 13 transformation files
- Stage 5: 13 model files + metrics
- Stage 3: aggregated_features.csv

**Key Functions**:
- `validate_stage_dependencies()` - Pre-flight (line 70)
- `generate_video_rf_json()` - Video RF analysis (line 165)
- `generate_window_rf_json()` - Window RF analysis (line 289)
- `generate_window_kmeans_json()` - Window K-Means (line 420)

**Checkpoint**: `{bucket_path}/checkpoints/stage_6_checkpoint.json`

**Depends On**: Stages 3, 4, 5 (all ML artifacts)

**Consumed By**: Stage 7 (LLM analysis)

**Error Strategy**:
- **FileNotFoundError/ValueError** → Skip bucket
- **RuntimeError** (API issues) → Exit pipeline

**Post-Execution Validation**: JSON structure, cluster count, feature importance checks

---

### Stage 7: LLM Analysis (Hybrid Two-Phase)

**Implementation**: [`ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py`](ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py)
**Entry Point**: `stage7_llm_analysis_main()` (orchestrator wrapper)
**Orchestrator Call**: [`rumiai_ml_batch.py:1751-1900`](rumiai_ml_batch.py#L1751-L1900)

**Inputs**:
- Stage 6: All 13 ML analysis JSONs
- Stage 2.7: Content classification files
- Environment: `ANTHROPIC_API_KEY`

**Outputs**:
```
{bucket_path}/ml_analysis/llm/
├── hook_analysis.json                # Phase 1: Window-level insights
├── middle_1_analysis.json
├── middle_2_analysis.json
├── middle_3_analysis.json
├── closing_analysis.json
├── complete_analysis_{bucket}.json   # Phase 2: Bucket summary
├── winning_formulas.json             # Phase 2: Creative templates (for Stage 8)
└── .phase1_status.json               # Checkpoint file
```

**Phase 1: Window-Level Analysis** (Parallel)
- Analyzes each window individually (hook, middle_X, closing)
- Combines K-Means clusters + RF feature importance
- Generates creative insights per window
- **API Config**: Claude Sonnet 4, 4000 tokens, 0.3 temperature, 90s timeout
- **Retry Strategy**: 3 attempts with exponential backoff (0s, 2s, 4s)

**Phase 2: Cross-Window Synthesis** (Sequential)
- Synthesizes insights across all windows
- Creates creative formulas (templates)
- Generates `winning_formulas.json` for Stage 8
- **API Config**: Claude Sonnet 4, 8000 tokens, 0.4 temperature, 180s timeout

**Key Functions**:
- `run_phase1_parallel()` - Window analyses (line 95)
- `analyze_window_with_retry()` - Single window with retry (line 224)
- `run_phase2_synthesis()` - Cross-window synthesis (line ~450)

**Checkpoint**: `.phase1_status.json` (tracks completed windows, enables resume)

**Checkpoint Strategy**: **IMPLICIT** - Uses output file existence, not explicit checkpoint file

**Depends On**: Stage 6 (ML analysis JSONs)

**Consumed By**: Stage 8 (report generation)

**Error Strategy**:
- **FileNotFoundError/ValueError** → Skip bucket
- **RuntimeError** (API auth) → Exit pipeline

**Validation**: JSON structure validation, cluster count checks

---

### Stage 8: Report Generation (PLANNED)

**Implementation**: 📝 **NOT YET IMPLEMENTED** - See [`documentation_migration/FutureDevelopments/Stage8MVP2.md`](documentation_migration/FutureDevelopments/Stage8MVP2.md)

**Entry Point**: TBD (4 separate extraction scripts)

**Inputs**:
- Stage 1: `winner_analysis.json`, `selection_manifest.json`
- Stage 2.7: `content_analysis/validated/bucket_{name}/*_content.json`
- Stage 7: `ml_analysis/llm/winning_formulas.json`

**Outputs** (4 Report Types):

**Report 1: Hashtag → Client**
```
{analysis_base}/
└── {target}_client_data.xlsx         # Business metrics, aggregated across all buckets
```

**Report 2: Hashtag → Creator**
```
{analysis_base}/
├── {target}_creator_data.xlsx        # 3 tabs (one per winning bucket formula)
└── qr_codes/
    ├── {target}_{bucket1}_top.png
    ├── {target}_{bucket1}_bottom.png
    ├── {target}_{bucket2}_top.png
    ├── {target}_{bucket2}_bottom.png
    ├── {target}_{bucket3}_top.png
    └── {target}_{bucket3}_bottom.png
```

**Report 3: Single Competitor**
```
/data/clients/{client}/competitors/{competitor}/top_contrastive/
├── {competitor}_analysis_data.xlsx   # Competitor insights, aggregated
└── qr_codes/
    └── {competitor}_top.png
```

**Report 4: Multi-Competitor**
```
/data/clients/{client}/market_intelligence/multi_competitor/
├── market_intelligence_report.xlsx   # Market comparison
└── qr_codes/
    ├── {competitor1}_top.png
    ├── {competitor2}_top.png
    └── {competitor3}_top.png
```

**Extraction Scripts** (4 separate CLI tools):
- `extract_client_data.py` - Report 1
- `extract_creator_data.py` - Report 2
- `extract_competitor_data.py` - Report 3
- `extract_multi_competitor_data.py` - Report 4

**Depends On**:
- Reports 1, 3, 4: Stages 1, 2.7
- Report 2: Stages 1, 2.7, 7 (winning_formulas.json)

**Note**: Stage 8 does NOT require Stages 3-6 (ML training) for report generation

---

## File Lifecycle Map

### Critical Files Created Per Stage

| File | Created By | Consumed By | Lifespan | Location | Schema Doc |
|------|------------|-------------|----------|----------|------------|
| `winner_analysis.json` | Stage 1 | Stage 2.5, 8 | Pipeline | `{analysis_base}/` | Top 3 buckets by video count |
| `selection_manifest.json` | Stage 2.5 | Stage 2.5.1, 2.6, 2.7, 8 | Pipeline | `{analysis_base}/` | 120 video IDs split by bucket+performer |
| `selected_videos.json` | Stage 1 | Stage 2, 8 | Pipeline | `{bucket_path}/` | TikTok API metadata (40 videos) |
| `{video_id}_temporal_windows_updated.json` | Stage 2 | Stage 2.5 → 3 | Pipeline | Stage 2: `/home/jorge/rumiaifinal/insights/`<br>Stage 2.5+: `{bucket_path}/analysis/insights/` | 9 ML services aggregated |
| `{hashtag}_raw_discovery.json` | Stage 2.6 | Stage 2.6 (manual curation) | Persistent | `{analysis_base}/content_taxonomies/` | LLM-generated taxonomy (7 categories) |
| `{hashtag}_taxonomy.json` | Stage 2.6 (manual) | Stage 2.7 | Persistent | `{analysis_base}/content_taxonomies/` | Manually curated taxonomy |
| `{video_id}_content.json` | Stage 2.7 | Stage 7, 8 | Pipeline | `{analysis_base}/content_analysis/validated/bucket_{name}/` | 15 classification fields |
| `aggregated_features.csv` | Stage 3 | Stage 4, 6 | Pipeline | `{bucket_path}/ml_analysis/` | 350+ features × 40 videos |
| `rf_transformed.csv` | Stage 4 | Stage 5, 6 | Pipeline | `{bucket_path}/ml_analysis/` | Video-level RF (147 features) |
| `{window}_scalers.pkl` | Stage 4 | Stage 5 (training) | Pipeline | `{bucket_path}/ml_analysis/` | MinMaxScaler objects |
| `rf_video_{bucket}.pkl` | Stage 5 | Stage 6 | Pipeline | `{bucket_path}/models/` | Trained Random Forest |
| `{window}_kmeans_{bucket}.pkl` | Stage 5 | Stage 6 | Pipeline | `{bucket_path}/models/` | Trained K-Means |
| `model_metrics.json` | Stage 5 | Stage 6 validation | Pipeline | `{bucket_path}/models/` | Performance metrics |
| `{window}_rf_analysis.json` | Stage 6 | Stage 7 | Pipeline | `{bucket_path}/ml_analysis/` | Feature importance (top 10) |
| `{window}_kmeans_analysis.json` | Stage 6 | Stage 7 | Pipeline | `{bucket_path}/ml_analysis/` | Cluster analysis (3 clusters) |
| `winning_formulas.json` | Stage 7 | Stage 8 (Report 2) | Persistent | `{bucket_path}/ml_analysis/llm/` | Creative templates |
| `complete_analysis_{bucket}.json` | Stage 7 | Stage 8 | Persistent | `{bucket_path}/ml_analysis/llm/` | Bucket summary |

**Lifespan Types**:
- **Pipeline**: Temporary, used during ML pipeline execution only
- **Persistent**: Kept for future reference, used by reporting

### Critical Path Dependencies

```
selection_manifest.json (Stage 2.5)
    ├─→ Stage 2.5.1 (validation)
    ├─→ Stage 2.6 (discovery)
    ├─→ Stage 2.7 (classification)
    └─→ Stage 8 (all reports)

{hashtag}_taxonomy.json (Stage 2.6 manual)
    └─→ Stage 2.7 (BLOCKS until curated)

validated/*_content.json (Stage 2.7)
    ├─→ Stage 7 (LLM context)
    └─→ Stage 8 (all reports)

winning_formulas.json (Stage 7)
    └─→ Stage 8 Report 2 ONLY
```

---

## Checkpoint Strategy

### Stages with Explicit Checkpoints

| Stage | Checkpoint File | Schema | Skip Logic |
|-------|----------------|--------|------------|
| **Stage 1** | `checkpoints/stage_1_checkpoint.json` | `{status, winning_buckets, output_files, timestamp}` | Validates schema + file existence |
| **Stage 2** | `{bucket_path}/checkpoints/stage_2_checkpoint.json` | Per-bucket tracking | Re-runs if checkpoint corrupt |
| **Stage 3** | `{bucket_path}/checkpoints/stage_3_checkpoint.json` | `{status: "completed"}` | Validates status field |
| **Stage 4** | `{bucket_path}/checkpoints/stage_4_checkpoint.json` | Output file list + counts | Fallback: CSV existence check |
| **Stage 5** | `{bucket_path}/checkpoints/stage_5_checkpoint.json` | Model file list | Validates model count |
| **Stage 6** | `{bucket_path}/checkpoints/stage_6_checkpoint.json` | JSON file list | Validates 13 files |

### Stages with Implicit Checkpoints

| Stage | Checkpoint Method | Resume Strategy |
|-------|-------------------|-----------------|
| **Stage 2.7** | `.checkpoints/classification_checkpoint.json` | Thread-safe per-video tracking |
| **Stage 7** | Output file existence (`complete_analysis_{bucket}.json`) | Re-runs if file missing |
| **Stage 7 Phase 1** | `.phase1_status.json` | Tracks completed windows, resumes incomplete |

### Checkpoint Validation Flow

```python
# Example from rumiai_ml_batch.py Stage 1 checkpoint validation

checkpoint_path = analysis_base / "checkpoints" / "stage_1_checkpoint.json"

if checkpoint_path.exists():
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    # Schema validation
    if checkpoint.get("status") == "completed":
        # Check output files exist
        if all(Path(f).exists() for f in checkpoint.get("output_files", [])):
            logger.info("Stage 1 checkpoint valid - skipping")
            continue
        else:
            logger.warning("Stage 1 checkpoint invalid - re-running")
            checkpoint_path.unlink()
```

---

## Error Propagation Matrix

### Error Handling Strategy by Exception Type

| Exception Type | Stage Action | Pipeline Action | Exit Code | Rationale |
|---------------|--------------|-----------------|-----------|-----------|
| **ValueError** | Skip bucket | Continue | 1 | Bucket-specific data issue |
| **AssertionError** | Skip bucket | Continue | 3 | Output validation failed |
| **FileNotFoundError** | Skip bucket | Continue | 1 | Missing input (upstream failure) |
| **IOError / OSError** | None | **Exit pipeline** | 4 | System-wide issue (disk full) |
| **TimeoutError** | None | **Exit pipeline** | 8 | System overload |
| **RuntimeError** (API) | None | **Exit pipeline** | 99 | Authentication failure |
| **StageInputError** | Skip bucket | Continue | 1 | Custom: Stage input missing |
| **InsufficientDataError** | Skip bucket | Continue | 1 | Custom: Below min threshold |
| **ModelTrainingError** | Skip bucket | Continue | 1 | Custom: Training failed |
| **ValidationError** | Skip bucket | Continue | 3 | Custom: Validation failed |

### Per-Stage Error Behavior

| Stage | Skip Bucket (Continue) | Exit Pipeline (Stop) |
|-------|------------------------|----------------------|
| **Stage 1** | Never | All errors |
| **Stage 2** | Processing errors | IOError, TimeoutError |
| **Stage 2.5** | Missing files | IOError |
| **Stage 2.5.1** | Never | Threshold failure (<30 valid) |
| **Stage 2.6** | Never | One-time execution, blocks until manual curation |
| **Stage 2.7** | Per-video errors | API authentication |
| **Stage 3** | ValueError, AssertionError | IOError, TimeoutError |
| **Stage 4** | ValueError, AssertionError | IOError, TimeoutError |
| **Stage 5** | Custom exceptions | IOError |
| **Stage 6** | FileNotFoundError, ValueError | RuntimeError (API) |
| **Stage 7** | FileNotFoundError, ValueError | RuntimeError (API auth) |

### Exit Codes Reference

```python
# From rumiai_ml_batch.py lines 1958-end

EXIT_CODES = {
    0: "Success (full pipeline completion)",
    1: "Error (validation failure, missing inputs)",
    2: "Paused for manual curation (Stage 2.6 complete)",
    3: "Assertion error (output validation failed)",
    4: "I/O failure (disk full, permissions)",
    8: "Timeout (processing exceeded limits)",
    99: "Unexpected error",
    130: "User interrupt (Ctrl+C)"
}
```

### Cross-Stage Impact Matrix

| Stage Modified | Impacts Downstream | Must Re-run | Auto-Detected? |
|---------------|-------------------|-------------|----------------|
| **Stage 1 (re-scrape)** | All stages (2-8) | All downstream | No - manual cleanup |
| **Stage 2 (fix ML)** | Stages 2.5-7 | All downstream | No - cached temporal_windows |
| **Stage 2.5 (re-organize)** | Stages 2.6, 2.7, 3-7 | All downstream | Yes - manifest timestamp |
| **Stage 2.6 (edit taxonomy)** | Stage 2.7, 7, 8 | Classification + analysis | Yes - taxonomy version |
| **Stage 2.7 (re-classify)** | Stages 7, 8 | Analysis + reports | Yes - checkpoint tracks |
| **Stage 3 (re-aggregate)** | Stages 4-7 | ML training pipeline | Yes - checkpoint tracks |
| **Stage 4 (re-transform)** | Stages 5-7 | Training + analysis | Yes - checkpoint tracks |
| **Stage 5 (re-train)** | Stages 6-7 | Analysis only | Yes - checkpoint tracks |
| **Stage 6 (re-analyze)** | Stage 7 | LLM analysis | Yes - checkpoint tracks |
| **Stage 7 (re-analyze)** | Stage 8 | Reports only | Yes - output file existence |

---

## Implementation Documentation

For detailed implementation guides on specific stages, see:

- **Stage 1**: [`docs/stages/STAGE_1_IMPL.md`](docs/stages/STAGE_1_IMPL.md) (Video Discovery)
- **Stage 2**: [`docs/stages/STAGE_2_IMPL.md`](docs/stages/STAGE_2_IMPL.md) (ML Processing)
- **Stage 2.5**: [`docs/stages/STAGE_2.5_IMPL.md`](docs/stages/STAGE_2.5_IMPL.md) (File Organization)
- **Stage 2.6**: [`docs/stages/STAGE_2.6_IMPL.md`](docs/stages/STAGE_2.6_IMPL.md) (Content Discovery)
- **Stage 2.7**: [`docs/stages/STAGE_2.7_IMPL.md`](docs/stages/STAGE_2.7_IMPL.md) (Content Classification)
- **Stage 3**: [`docs/stages/STAGE_3_IMPL.md`](docs/stages/STAGE_3_IMPL.md) (Feature Aggregation)
- **Stage 4**: [`docs/stages/STAGE_4_IMPL.md`](docs/stages/STAGE_4_IMPL.md) (Feature Transformation)
- **Stage 5**: [`docs/stages/STAGE_5_IMPL.md`](docs/stages/STAGE_5_IMPL.md) (Model Training)
- **Stage 6**: [`docs/stages/STAGE_6_IMPL.md`](docs/stages/STAGE_6_IMPL.md) (ML Analysis Generation)
- **Stage 7**: [`docs/stages/STAGE_7_IMPL.md`](docs/stages/STAGE_7_IMPL.md) (LLM Analysis)
- **Stage 8**: [`docs/stages/STAGE_8_IMPL.md`](docs/stages/STAGE_8_IMPL.md) (Report Generation - PLANNED)

---

## Related Documentation

- **Quick Reference**: [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) - Orientation guide
- **Business Context**: [`BusinessContext.md`](BusinessContext.md) - Why RumiAI exists
- **System Architecture**: [`SystemArchitecturev2.md`](SystemArchitecturev2.md) - Technical deep dive
- **ML Roadmap**: [`MLROADMAP.md`](MLROADMAP.md) - ML training vision
- **Stage 8 Specification**: [`documentation_migration/FutureDevelopments/Stage8MVP2.md`](documentation_migration/FutureDevelopments/Stage8MVP2.md)
- **File Schemas**: [`docs/schemas/`](docs/schemas/) - File format specifications
- **TI Documentation**: [`documentation_migration/`](documentation_migration/) - Technical implementation specs

---

## Usage Examples

### For LLM Agents: Fix Bug in Specific Stage

**Scenario**: Fix bug in Stage 3 aggregation - missing cross-window features

**Agent Workflow**:
1. Read this file → Locate Stage 3 section
2. Click link to [`docs/stages/STAGE_3_IMPL.md`](docs/stages/STAGE_3_IMPL.md)
3. Read implementation details (entry point, functions, line numbers)
4. Read specific files mentioned in STAGE_3_IMPL.md
5. Make changes
6. Follow "Debugging Checklist" from STAGE_3_IMPL.md

### For LLM Agents: Add New Stage

**Scenario**: Implement Stage 8.2 (Extract Creator Data)

**Agent Workflow**:
1. Read this file → Understand pipeline architecture
2. Identify dependencies (Stage 8 requires Stages 1, 2.7, 7)
3. Read [`documentation_migration/FutureDevelopments/Stage8MVP2.md`](documentation_migration/FutureDevelopments/Stage8MVP2.md)
4. Create `docs/stages/STAGE_8_IMPL.md` (use STAGE_7_IMPL.md as template)
5. Update this file (PRODUCTION_FLOW.md) with Stage 8 contract
6. Implement extraction script
7. Update orchestrator if integration needed

### For LLM Agents: Trace Data Flow

**Scenario**: Where does `selection_manifest.json` come from and who uses it?

**Agent Workflow**:
1. Read this file → File Lifecycle Map table
2. Find `selection_manifest.json` row:
   - **Created By**: Stage 2.5
   - **Consumed By**: Stages 2.5.1, 2.6, 2.7, 8
3. Read Stage 2.5 contract for creation details
4. Read consuming stages for usage patterns
5. Check schema: "120 video IDs split by bucket+performer"

---

## Maintenance Notes

### Updating This Document

**When to update PRODUCTION_FLOW.md**:
- Adding/removing pipeline stages
- Changing stage dependencies
- Modifying checkpoint strategies
- Adding new critical files
- Changing error handling strategies

**What NOT to update here**:
- Implementation details (goes in STAGE_*_IMPL.md)
- Function signatures (goes in STAGE_*_IMPL.md)
- Bug fixes (update implementation docs only)
- Configuration changes (update config docs)

### Document Hierarchy

```
PRODUCTION_FLOW.md (this file)
    ↓ links to
docs/stages/STAGE_*_IMPL.md
    ↓ links to
documentation_migration/*TI.md (Technical Implementation specs)
```

**Purpose of Each Layer**:
1. **PRODUCTION_FLOW.md**: Executive map (architecture, dependencies, contracts)
2. **STAGE_*_IMPL.md**: Implementation guide (entry points, functions, debugging)
3. **TI docs**: Deep technical specs (algorithms, formulas, edge cases)

---

**Document Version**: 1.0
**Generated**: 2025-01-28
**Source**: Systematic analysis of RumiAI codebase (Stages 1-8)
**Maintainer**: Update when pipeline architecture changes
