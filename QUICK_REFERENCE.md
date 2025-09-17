# RumiAI Quick Reference Guide

## 🎯 What is RumiAI?

**Current System (Production - v2.1)**: 
Zero-cost TikTok video analyzer that extracts professional insights using ML models and Python processing.
rumiai_v2/processors/temporal_compute.py is the main flow we use. 
We are deleting precompute_functions


## 📚 Documentation Reading Order

### For Understanding Future Development (20 min)
1. **postrefactorflow.md** - Explanation of the new architecture and flow 
2. **MLMVP2.md** - Feature engineering architecture for ML training
3. **MLProjectsGrassrootsv2.md** - Full ML training pipeline implementation plan
4. **DONE - refactortemporal.md** - Recent refactor we did with now correct flow for our main script python3 scripts/rumiai_runner.py 

## Future Plans
1. **MLProjectsGrassrootsv2.md** - General Plan
2. **MLMVP2.md** - Future ML training layer will identify viral patterns
1. **ImprovementsMLMVP.md** - This is a list of modifications we are currently making

## 🚀 Key System Facts

### Current Capabilities
- **Cost**: $0.00 per video (no API usage)
- **Speed**: ~2 Minutes
- **Output**: 1 JSON output per video
- **Format**: 1 ML Ready format which we still need to transform to RF and KMeans
- **ML Models**: YOLO, Whisper, MediaPipe, OCR, Scene Detection
- **Success Rate**: 100% (fail-fast architecture)


## 🔧 Common Commands

```bash
# Analyze a video (current system)
python3 scripts/rumiai_runner.py "VIDEO_URL"

# Test Flow
python test_temporal_compute_v2.py "VIDEO ID"

# Check configuration (hardcoded for Python-only)
cat rumiai_v2/config/settings.py | grep "use_python_only"
```

## ⚠️ Important Notes

1. **All settings are hardcoded** - No environment variables needed
2. **Fail-fast architecture** - Either complete success or immediate failure
3. **No Claude API** - 100% Python-only processing
4. **Real ML models** - Not mocked, actual YOLO/Whisper/MediaPipe


## 📖 Further Reading

### Technical Deep Dives
- Individual service docs: `ScenePacing.md`, `VisualOverlay.md`, etc.

### Architecture Philosophy
- Why Python-only: See RUMIAI_CORE_ARCHITECTURE_PRINCIPLES.md
- Why these ML models: See MLMVP2.md Section 2

---

*This guide provides entry point for understanding RumiAI's current capabilities and future direction. Start with the documentation reading order above for comprehensive understanding.*