# RumiAI Quick Reference Guide

## 🎯 What is RumiAI?

**Current System**: TikTok video analyzer that extracts 60+ ML features through temporal window analysis.
- **Core**: Processes videos through 9 ML services sequentially
- **Output**: JSON with ~20 features per temporal window for pattern detection
- **Purpose**: Internal tool for Tumi Labs' RippleOS consultancy to identify viral creative patterns

## 📚 START HERE - Documentation Reading Order

### For Fresh CLI Instance:
1. **[BusinessContext.md](./BusinessContext.md)** (1-2 min) - Why RumiAI exists, business problem, stakeholders
2. **[SystemArchitecturev2.md](./SystemArchitecturev2.md)** (10-15 min) - Technical architecture, data flow, services
3. **[MLROADMAP.md](./MLROADMAP.md)** (5 min) - Future ML pipeline development plans

### For Deep Technical Dives:
- **TotalFeatures.md**: `/documentation_migration/services/TotalFeatures.md` - All 20+ features explained
- **Service docs**: See `/documentation_migration/services/*.md` for individual service details

## 🚀 Key System Facts

### Current Capabilities
- **Processing Time**: ~60-80 seconds for a 60-second video
- **Output**: `temporal_windows_updated.json` with 60+ features per temporal window
- **Window Structure CODE**:
  | Duration | Middle Segments | Total Windows (Hook + Middle + Closing) |
  |----------|-----------------|-----------------------------------------|
  | 0-9s     | None (null)     | 2 (Hook + Closing only)                 |
  | 9-18s    | 3               | 5 (Hook + 3 Middle + Closing)           |
  | 18-33s   | 4               | 6 (Hook + 4 Middle + Closing)           |
  | 33-75s   | 5               | 7 (Hook + 5 Middle + Closing)           |
  | >75s     | 5 (capped)      | 7 (Hook + 5 Middle + Closing)           |

  **Window Structure ML Buckets** 
  ML Buckets - Total 8 - (Downstream - Training)
  | Bucket  | Duration | Window Structure Used |
  |---------|----------|-----------------------|
  | 0-3s    | 0-3s     | 1-window (Hook)       |
  | 3-9s    | 3-9s     | 2-window              |
  | 9-13s   | 9-13s    | 5-window              |
  | 13-18s  | 13-18s   | 5-window              |
  | 18-33s  | 18-33s   | 6-window              |
  | 33-60s  | 33-60s   | 7-window              |
  | 60-90s  | 60-90s   | 7-window              |
  | 90-120s | 90-120s  | 7-window              |

  - **Critical**: Videos ≤9s return middle_segments as null (not empty array)
  - **ML Note**: ML training will split 9-18s into two buckets (9-13s, 13-18s) for variance handling
  - **Segment Duration Calculation**: (Total Duration-(Hook+Closing))/Duration Segments
- **ML Services**: 9 services - YOLO, Whisper, MediaPipe, OCR, Scene Detection, FEAT, DeepFace, Audio Energy, Hashtag Analysis
- **Processing Modes**: Sequential (default) or Parallel
- **Architecture**: Self-contained services with fail-fast validation and checkpoint/resume


## 🔧 Common Commands

```bash
# Process a single video
python rumiai_runner.py "https://tiktok.com/@user/video/123"

# Set parallel processing mode (faster for short videos)
export PARALLEL_MODE=true

# Set sequential mode (default, better for long videos)
export PARALLEL_MODE=false

# View output structure
cat insights/[video_id]_temporal_windows_updated.json | jq '.temporal_windows | keys'

# Check what features are in each window
cat insights/[video_id]_temporal_windows_updated.json | jq '.temporal_windows.hook | keys'
```

## 🔧 Environment Variables
- `PARALLEL_MODE`: true/false (processing mode)
- `WHISPER_THREADS`: 1-16 (optimal: 4)
- `CV2_THREADS`: 1-8 (optimal: 2)
- `OMP_NUM_THREADS`: 1-8 (optimal: 2)

## 📁 Key File Locations

```
/home/jorge/rumiaifinal/
├── rumiai_runner.py                    # Main entry point
├── video_analyzer.py                   # Core orchestrator
├── timeline_builder.py                 # Temporal aggregation
├── BusinessContext.md                  # Business context
├── SystemArchitecturev2.md            # Technical architecture
├── MLROADMAP.md                        # Future ML pipeline
├── config/
│   └── bucket_definitions.py          # Shared bucket window configurations
├── documentation_migration/
│   └── services/
│       └── TotalFeatures.md           # All 60+ features explained
└── services/                           # Individual ML service modules
```

## 📊 For ML Development
See [MLROADMAP.md](./MLROADMAP.md) for upcoming ML training pipeline that will:
- Process 300 videos per hashtag
- Train Random Forest + K-means models per duration bucket
- Generate creative strategy reports for content creators