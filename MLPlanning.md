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

