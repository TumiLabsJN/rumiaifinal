# RumiAI Quick Reference Guide

## 🎯 What is RumiAI?

**Current System**: TikTok video analyzer that extracts ML features through temporal window analysis.
- **Core**: `rumiai_v2/processors/temporal_compute.py` - Computes features across hook/middle/closing windows
- **Output**: JSON with 50+ features per temporal window for ML training

## 📚 Understanding the System

### For Fresh CLI Instance (Start Here):
1. **This file** - Quick operational overview
2. **SystemArchitecture.md** - Complete technical flow and dependencies
3. **Sample output**: `/insights/[video_id]_temporal_windows_updated.json` - See actual output

### For Deep Technical Dives:
- **Services**: See `/documentation_migration/services/*.md` (Phase 1 docs)
- **Features**: See `/documentation_migration/features/*.md` (Phase 2 docs)

### For Future ML Development:
1. **MLMVP2.md** - Feature engineering architecture for ML training
2. **MLProjectsGrassrootsv2.md** - Full ML training pipeline implementation plan
3. **ImprovementsMLMVP.md** - Current feature improvements in progress

## 🚀 Key System Facts

### Current Capabilities
- **Processing Time**: ~60-80 seconds per video
- **Output**: Temporal windows JSON with 50+ features
- **Window Structure**: Hook (0-3s), Middle segments (~7.6s each), Closing (last 3s)
- **ML Services**: YOLO, Whisper, MediaPipe, OCR, Scene Detection, FEAT, DeepFace, Audio Energy
- **Architecture**: Self-contained services with fail-fast validation


## 🔧 Common Commands

```bash
# Analyze a video
python3 scripts/rumiai_runner.py "VIDEO_URL"

# Test temporal compute directly
python test_temporal_compute_v2.py

# View output structure
cat insights/[video_id]_temporal_windows_updated.json | jq '.temporal_windows | keys'

# Check what features are in each window
cat insights/[video_id]_temporal_windows_updated.json | jq '.temporal_windows.hook | keys'
```

## ⚠️ Important Notes

1. **Temporal Windows** - All features computed over hook/middle/closing segments
2. **Fail-fast validation** - Services validate data integrity before processing
3. **Self-contained services** - Each ML service can run independently
4. **Unified timeline** - All ML outputs merged into single timeline before temporal compute

## 📁 Key File Locations

```
/rumiai_v2/
├── scripts/rumiai_runner.py         # Main entry point
├── processors/
│   ├── temporal_compute.py          # Core temporal window processing
│   ├── timeline_builder.py          # ML data unification
│   └── video_analyzer.py            # Service orchestration
├── api/ml_services_unified.py       # ML service implementations
└── insights/                         # Output JSONs
```

---

*For comprehensive understanding, read SystemArchitecture.md. For specific components, see the Phase 1-2 documentation.*