ML next Phase


We are in the last phase of RumiAI

# Basics

## A. System Architecture

### Primary Goals
1. **Batch Video Analysis**
   - Process up to 300 videos sequentially through `rumiai_runner.py`
   - Implement checkpoint/resume system for failure recovery

2. **Client-Centric Data Organization**
   - Multi-tenant data structure: Client → (Hashtags | Competitors | Creators) → Duration Buckets → Videos
   - Three analysis types per client: Market research (hashtags), competitive intelligence (competitors), creator vetting (creators)
   - Bucket-specific analysis within client boundaries
   - Persistent client configuration management

3. **Duration-Specific ML Pattern Recognition**
   - Train **separate ML models (RF and Kmeans) for each duration bucket**
   - Recognize that 15-second patterns differ completely from 60-second patterns

4. **Creative Report Generation**
   - **Hashtags**: 5 strategy reports per bucket (40 total per hashtag) - "What works for #nutrition"
   - **Competitors**: 5 strategy reports per bucket (40 total per competitor) - "What @rival is doing"
   - **Creators**: Style profile + compatibility scoring - "Does this creator fit our brand?"
   - Multiple perspectives and strategies per duration bucket
   - Include bucket performance metrics for strategic content planning

5. **Creator Compatibility Analysis**
   - Analyze potential affiliate creators' natural content style (via their TikTok handle)
   - Cross-check creator features against client hashtag/competitor viral patterns
   - Generate compatibility scores per duration bucket (e.g., "Creator's 15s style matches #nutrition patterns: 82%")
   - Provide hiring recommendations: which creators naturally produce content that aligns with what works
   - **See detailed implementation**: [MLCreatorMatch.md](./documentation_migration/FutureDevelopments/MLCreatorMatch.md) 

## Client Architecture & Storage

### Data Example

#### ML Production Pipeline Architecture
```
/data/
├── clients/
│   ├── {client_id}/
│   │   ├── hashtags/                       # Hashtag-based analyses
│   │   │   └── {hashtag_name}/             # e.g., "nutrition", "fitness"
│   │   │       ├── buckets/                # Duration-based ML buckets (8 total)
│   │   │       │   ├── bucket_0-3s/
│   │   │       │   │   ├── videos/         # Raw MP4s for this duration (30-day retention)
│   │   │       │   │   ├── analysis/       # RumiAI outputs for videos in this bucket
│   │   │       │   │   │   ├── insights/   # temporal_windows JSON (1 per video)
│   │   │       │   │   │   ├── unified/    # Intermediate timeline+ml_data (debugging)
│   │   │       │   │   │   └── service_debug/  # emotion_detection, audio_energy outputs
│   │   │       │   │   ├── models/         # Trained models for THIS bucket
│   │   │       │   │   │   ├── random_forest_v1.pkl
│   │   │       │   │   │   ├── kmeans_v1.pkl
│   │   │       │   │   │   └── model_metrics.json
│   │   │       │   │   ├── reports/        # Bucket-specific creative reports (5 per bucket)
│   │   │       │   │   ├── checkpoints/    # Processing state for this bucket
│   │   │       │   │   └── logs/           # Bucket processing logs
│   │   │       │   │
│   │   │       │   ├── bucket_3-9s/        # Bucket 2
│   │   │       │   ├── bucket_9-13s/       # Bucket 3
│   │   │       │   ├── bucket_13-18s/      # Bucket 4
│   │   │       │   ├── bucket_18-33s/      # Bucket 5
│   │   │       │   ├── bucket_33-60s/      # Bucket 6
│   │   │       │   ├── bucket_60-90s/      # Bucket 7
│   │   │       │   └── bucket_90-120s/     # Bucket 8
│   │   │       │
│   │   │       └── hashtag_summary/        # Cross-bucket executive reports
│   │   │           ├── executive_report.pdf    # Bird's eye view for client
│   │   │           └── hashtag_metrics.json    # Aggregated stats across all buckets
│   │   │
│   │   ├── competitors/                    # Competitor tracking (what rivals are doing)
│   │   │   └── {competitor_handle}/        # e.g., "@rival_brand"
│   │   │       ├── buckets/                # Same 8-bucket structure
│   │   │       │   ├── bucket_0-3s/        # (full structure same as hashtags)
│   │   │       │   ├── bucket_3-9s/
│   │   │       │   ├── bucket_9-13s/
│   │   │       │   ├── bucket_13-18s/
│   │   │       │   ├── bucket_18-33s/
│   │   │       │   ├── bucket_33-60s/
│   │   │       │   ├── bucket_60-90s/
│   │   │       │   └── bucket_90-120s/
│   │   │       │
│   │   │       └── competitor_summary/     # Cross-bucket competitor analysis
│   │   │           ├── competitor_report.pdf   # "What @rival_brand is doing"
│   │   │           └── competitor_metrics.json
│   │   │
│   │   └── creators/                       # Creator vetting (potential affiliates)
│   │       └── {creator_handle}/           # e.g., "@potential_affiliate"
│   │           ├── buckets/                # Same 8-bucket structure
│   │           │   ├── bucket_0-3s/        # (full structure same as hashtags)
│   │           │   ├── bucket_3-9s/
│   │           │   ├── bucket_9-13s/
│   │           │   ├── bucket_13-18s/
│   │           │   ├── bucket_18-33s/
│   │           │   ├── bucket_33-60s/
│   │           │   ├── bucket_60-90s/
│   │           │   └── bucket_90-120s/
│   │           │
│   │           └── creator_summary/        # Cross-check analysis
│   │               ├── style_profile.pdf       # Creator's natural content style
│   │               ├── compatibility_report.pdf # Match vs client hashtags/competitors
│   │               ├── compatibility_scores.json # Numeric scores per bucket
│   │               └── creator_metrics.json
```

#### Development/Testing Architecture (Preserved)
```
/insights/                     # Final temporal windows JSON (single video CLI usage)
/unified_analysis/             # Intermediate timeline + ml_data
/temp/                         # Downloaded videos
/emotion_detection_outputs/    # Service debugging
/audio_energy_outputs/         # Service debugging
/logs/                         # Processing logs
```

**Note**: Root-level directories remain for `rumiai_runner.py` CLI development/testing. Production ML batch processing uses `/data/clients/` structure.


### Data Retention Policy
- **Raw Videos**: 30 days (then delete to save space, can re-download if needed)
- **ML Analysis**: 6 months (compressed after 30 days)
- **ML Models**: Keep latest 3 versions per client/hashtag
- **Reports**: Indefinite (small size, high value)
- **Checkpoints**: 7 days after successful completion

### Storage Cost Optimization
- **Video Deletion**: Remove raw videos after 30 days


## Flow

This section documents how to initialize ML batch processing via CLI for each analysis type.

**MVP Approach**: Command-line flags for all inputs (enables automation, scriptable, repeatable). Interactive mode and config files are future enhancements.

**Apify Parameters**: Exposed as CLI flags (`--video-count`, `--date-filter`, `--analysis-mode`) for full control per analysis.

**Analysis Modes**: Two modes enable different business questions - `top` (what works?) vs `recent` (what's happening now?). See [MLAnalysisMode.md](./documentation_migration/FutureDevelopments/MLAnalysisMode.md) for detailed implementation.

**Checkpoint/Resume**: Auto-resume on restart (detects checkpoint, continues automatically). Use `--force` flag to discard checkpoint and restart fresh. See [MLCheckpointResume.md](./documentation_migration/FutureDevelopments/MLCheckpointResume.md) for details.

---

### Hashtag Flow

**Purpose**: Analyze videos from a hashtag to identify viral creative patterns and train duration-specific ML models

**Default Mode**: `--analysis-mode top` (analyze highest-engagement videos)

**Command**:
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type hashtag \
  --target "#nutrition" \
  --video-count 300 \
  --date-filter "last_90_days" \
  --analysis-mode top
```

**Inputs**:
- `--client`: Client identifier (creates `/data/clients/{client_id}/`)
- `--analysis-type`: `hashtag` | `competitor` | `creator`
- `--target`: Hashtag name (e.g., `#nutrition`)
- `--video-count`: Number of videos to scrape (default: 300)
- `--date-filter`: Recency filter (e.g., `last_90_days`, `2024-01-01:2025-01-01`)
- `--analysis-mode`: `top` (default) | `recent`

**Process (Top Mode)**:
1. **Apify Scraping**: Fetch 300 videos for #nutrition sorted by engagement
2. **Client-side Filtering**: Apply date filter, remove duplicates
3. **Duration Bucketing**: Split videos into 8 duration buckets
4. **Video Selection**: Top 40 + Bottom 20 per bucket (contrastive analysis)
5. **Batch Analysis**: Run RumiAI on selected videos (sequential, with checkpoints)
6. **ML Training**: Train Random Forest + K-Means per bucket
7. **Report Generation**: Create 5 strategy reports per bucket (40 total)

**Process (Recent Mode)** - Optional for trend monitoring:
Same as Top Mode but sorts by publish date instead of engagement. Use case: "What are creators posting NOW?"

**Outputs**:
```
/data/clients/client_acme_corp/hashtags/nutrition/
  ├── buckets/
  │   ├── bucket_13-18s/
  │   │   ├── videos/          # 60 MP4 files (40 top + 20 bottom)
  │   │   ├── analysis/        # 60 temporal_windows JSONs
  │   │   ├── models/          # random_forest_v1.pkl, kmeans_v1.pkl
  │   │   └── reports/         # 5 PDF creative strategy reports
  │   └── ... (8 buckets total)
  └── hashtag_summary/
      └── executive_report.pdf  # Cross-bucket overview for client
```

**Duration**: ~6-8 hours for 300 videos (60-80s per video analysis)

---

### Competitor Flow

**Purpose**: Benchmark competitor's successful content or track their current strategy

**Default Mode**: `--analysis-mode top` (analyze highest-engagement videos)

**Command (Top Mode)** - Benchmark their best:
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode top
```

**Command (Recent Mode)** - Track current strategy:
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode recent
```

**Inputs**:
- `--client`: Client identifier
- `--analysis-type`: `competitor`
- `--target`: Competitor TikTok handle (e.g., `@rival_brand`)
- `--video-count`: Number of videos to scrape (default: 150)
- `--date-filter`: Recency filter
- `--analysis-mode`: `top` (default) | `recent`

**Process (Top Mode)** - "What works for rival?":
1. **Apify Scraping**: Fetch 150 videos from @rival_brand sorted by engagement
2. **Client-side Filtering**: Apply date filter
3. **Duration Bucketing**: Split videos into 8 duration buckets
4. **Batch Analysis**: Process ALL 150 videos (no top/bottom selection)
5. **Pattern Analysis**: Identify creative patterns that correlate with rival's success
6. **Report Generation**: "What works for @rival_brand" (patterns in their best content)

**Process (Recent Mode)** - "What is rival posting now?":
Same as Top Mode but sorts by publish date. Identifies current content strategy and potential shifts.

**Outputs**:
```
/data/clients/client_acme_corp/competitors/rival_brand/
  ├── videos/                # 150 MP4 files
  ├── analysis/              # 150 temporal_windows JSONs
  ├── distribution_analysis/
  │   └── duration_distribution.json  # What durations rival focuses on
  └── competitor_summary/
      └── competitor_report.pdf  # "What @rival_brand is doing"
```

**Duration**: ~3-4 hours for 150 videos (60-80s per video analysis)

**Key Differences vs Hashtag**:
- Hashtag: 300+ videos → 480 processed (top 40 + bottom 20 per bucket) - Market research
- Competitor: 150 videos → all processed - Competitive intelligence
- No contrastive analysis (no top/bottom selection within buckets)

---

### Creator Flow

**Purpose**: Vet potential affiliate creators by analyzing their natural content style and compatibility with client patterns

**Default Mode**: `--analysis-mode recent` (analyze most recent videos for natural style)

**Command (Recent Mode)** - Natural style vetting (primary use case):
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type creator \
  --target "@potential_affiliate" \
  --video-count 40 \
  --analysis-mode recent \
  --compare-to hashtag:nutrition
```

**Command (Top Mode)** - Peak performance analysis (optional):
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type creator \
  --target "@potential_affiliate" \
  --video-count 40 \
  --analysis-mode top \
  --compare-to hashtag:nutrition
```

**Inputs**:
- `--client`: Client identifier
- `--analysis-type`: `creator`
- `--target`: Creator TikTok handle (e.g., `@potential_affiliate`)
- `--video-count`: Number of videos to analyze (default: 40)
- `--analysis-mode`: `recent` (default) | `top`
- `--compare-to`: What to compare against (format: `hashtag:{name}` or `competitor:{handle}`)

**Process (Recent Mode)** - "What does creator naturally produce?":
1. **Apify Scraping**: Fetch most recent 40 videos from creator (sorted by date)
2. **Batch Analysis**: Run RumiAI on all 40 videos
3. **Duration Distribution**: Calculate % of content per bucket (natural style)
4. **Feature Aggregation**: Average features per bucket
5. **Compatibility Scoring**: Compare vs client's hashtag/competitor patterns
6. **Report Generation**: Style profile + compatibility report + hiring recommendation

**Process (Top Mode)** - "What does creator do when they go viral?":
Same as Recent Mode but analyzes top 40 videos by engagement. Use case: Understanding peak performance style for coaching purposes.

**Outputs**:
```
/data/clients/client_acme_corp/creators/potential_affiliate/
  ├── videos/                      # 40 MP4 files
  ├── analysis/insights/           # 40 temporal_windows JSONs
  ├── distribution_analysis/
  │   └── duration_distribution.json  # Natural duration preferences
  ├── compatibility_analysis/
  │   └── vs_hashtag_nutrition/
  │       ├── distribution_match.json      # How well durations align
  │       ├── feature_alignment.json       # Feature-level comparison
  │       ├── compatibility_score.json     # Final score (0-1)
  │       └── hiring_recommendation.json   # Tier 1-4 + reasoning
  └── creator_summary/
      ├── style_profile.pdf            # Creator's natural content style
      └── compatibility_report.pdf     # Match vs client patterns
```

**Duration**: ~53 minutes for 40 videos (80s per video)

**Key Differences vs Hashtag/Competitor**:
- Only 40 videos (snapshot for vetting, not comprehensive analysis)
- No ML model training (uses existing hashtag/competitor models for comparison)
- Outputs compatibility score + hiring recommendation (not creative strategy reports)
- Default mode is `recent` (natural style) vs `top` (best performance)

---

### Future Enhancements (Post-MVP)

#### Interactive Mode
For beginners, offer guided prompts instead of memorizing flags:
```bash
$ python rumiai_ml_batch.py --interactive
```

#### Configuration File Support
For complex multi-analysis batches:
```bash
python rumiai_ml_batch.py --config config/acme_nutrition_analysis.yaml
```

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

## 3. Analysis Mode System

**Purpose**: Enable dual-purpose analysis - understand "what works" (top-performing content) vs "what's happening now" (current trends/strategy).

**Key Capabilities**:
- `top` mode: Analyze highest-engagement videos to identify successful patterns
- `recent` mode: Analyze most recent videos to track current strategies and trends
- Apify integration for both engagement-based and date-based sorting
- Different default modes per analysis type (hashtag/competitor: top, creator: recent)

**Business Value**:
- Competitive intelligence: Benchmark rival's best work OR track their strategy shifts
- Trend detection: Identify market changes by analyzing recent vs historical patterns
- Creator vetting: Natural style (recent) vs peak performance (top) for different use cases

**Implementation Details**: See [MLAnalysisMode.md](./documentation_migration/FutureDevelopments/MLAnalysisMode.md)

**Priority**: HIGH - Core feature for flexible analysis across different business questions

