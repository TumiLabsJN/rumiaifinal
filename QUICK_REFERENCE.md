# RumiAI Quick Reference Guide

## 🎯 What is RumiAI?

**Current System (Production - v2.1)**: 
Zero-cost TikTok video analyzer that extracts professional insights using ML models and Python processing.

**Future System (In Development - ML Training)**:
Pattern recognition layer that identifies viral creative strategies by training on analyzed videos.

## 📚 Documentation Reading Order

### For Understanding Current System (45 min total)
1. **FlowStructure.md** (5 min) - Quick visual overview of the pipeline
2. **RUMIAI_CORE_ARCHITECTURE_PRINCIPLES.md** (10 min) - The "why" behind design decisions
3. **Codemappingfinal.md** (15 min) - Detailed file mapping and recent optimizations
4. **ML_DATA_PROCESSING_PIPELINE.md** (15 min) - Deep dive into data transformation

### For Understanding Future Development (20 min)
5. **MLMVP2.md** - Feature engineering architecture for ML training
6. **MLProjectsGrassrootsv2.md** - Full ML training pipeline implementation plan

## 🚀 Key System Facts

### Current Capabilities
- **Cost**: $0.00 per video (no API usage)
- **Speed**: ~53 seconds total processing (33% faster than v2.0)
- **Output**: 7-8 professional JSON analyses per video
- **Format**: 6-block CoreBlocks structure
- **ML Models**: YOLO, Whisper, MediaPipe, OCR, Scene Detection
- **Success Rate**: 100% (fail-fast architecture)

### Recent v2.1 Optimizations
- SharedAudioExtractor: 40% faster audio processing
- Adaptive OCR sampling: 50% reduction in processing
- Dead code removed: 595+ lines eliminated
- Bug fixes: Face detection 0% → 97%

## 🔄 Processing Pipeline

```
Current Flow (What We Have):
TikTok URL → Video Download → ML Analysis → Timeline Building → 
Python Processing → 6-Block JSONs → insights/{video_id}/

Future Flow (What We're Building):
6-Block JSONs → Feature Engineering → Canonical JSON → 
ML Training → Creative Pattern Recognition → Viral Strategy Reports
```

## 💡 Understanding the Architecture

### Core Principle
**Python-Only Processing**: Everything runs locally at zero cost, no external APIs.

### The 6-Block Professional Format
Every analysis outputs this structure:
1. **CoreMetrics** - Primary measurements
2. **Dynamics** - Temporal progressions
3. **Interactions** - Cross-modal relationships
4. **KeyEvents** - Peak moments and highlights
5. **Patterns** - Detected techniques
6. **Quality** - Confidence and reliability scores

### Why This Matters
- Current system analyzes WHAT is in videos
- Future ML layer will identify WHICH patterns go viral
- Together: Complete creative intelligence system

## 🎨 The Vision

### Today (v2.1 - Production)
- Analyze any TikTok video in 53 seconds
- Extract 150+ features at zero cost
- Professional quality output

### Tomorrow (ML Training - Development)
- Train on top/bottom performing videos
- Identify viral creative patterns by industry
- Generate actionable creative strategies
- Duration-specific recommendations (15s vs 60s content)

### Business Model
```
Brands (Pay) → Tumi Labs (Analyze) → Content Creators (Execute)
                     ↓
            ML-Driven Creative Strategies
```

## 📂 Output Structure

```
insights/
  {video_id}/
    creative_density/       # Element distribution analysis
    emotional_journey/      # Emotion progression tracking
    person_framing/        # Human presence analysis
    scene_pacing/          # Editing rhythm patterns
    speech_analysis/       # Audio and speech patterns
    visual_overlay/        # Text and overlay analysis
    metadata_analysis/     # Engagement metrics
    temporal_markers/      # Hook and retention analysis
```

## 🔧 Common Commands

```bash
# Analyze a video (current system)
python3 scripts/rumiai_runner.py "VIDEO_URL"

# Output location
ls insights/{video_id}/

# Check configuration (hardcoded for Python-only)
cat rumiai_v2/config/settings.py | grep "use_python_only"
```

## ⚠️ Important Notes

1. **All settings are hardcoded** - No environment variables needed
2. **Fail-fast architecture** - Either complete success or immediate failure
3. **No Claude API** - 100% Python-only processing
4. **Real ML models** - Not mocked, actual YOLO/Whisper/MediaPipe

## 🔮 Development Status

| Component | Status | Notes |
|-----------|--------|-------|
| Video Analysis (v2.1) | ✅ Production | 53s processing, $0.00 cost |
| ML Feature Engineering | 🔨 Development | Canonical JSON design complete |
| ML Training Pipeline | 📋 Planning | Random Forest + K-means approach |
| Creative Pattern Reports | 📋 Planning | Duration-specific insights |

## 📖 Further Reading

### Technical Deep Dives
- Individual service docs: `ScenePacing.md`, `VisualOverlay.md`, etc.
- ML feature analysis: `*MLA.md` files (e.g., `metadata_analysisMLA.md`)

### Architecture Philosophy
- Why Python-only: See RUMIAI_CORE_ARCHITECTURE_PRINCIPLES.md
- Why these ML models: See MLMVP2.md Section 2

---

*This guide provides entry point for understanding RumiAI's current capabilities and future direction. Start with the documentation reading order above for comprehensive understanding.*