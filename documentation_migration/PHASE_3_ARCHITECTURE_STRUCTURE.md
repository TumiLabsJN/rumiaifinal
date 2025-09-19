# PHASE 3: Architecture Documentation Structure

## ⚠️ CRITICAL CONTEXT
Phase 3 synthesizes everything from Phase 1 (services) and Phase 2 (features) into system-wide architecture and migration documentation.
This is the FINAL phase that creates the complete picture.

## 📋 Phase 3 Scope
Document the COMPLETE SYSTEM FLOW and MIGRATION STATUS.
No new technical details - just connecting everything documented in Phases 1 & 2.

## 📁 Documents to Create
1. **SystemArchitecture.md** - Complete data flow, dependencies, system design
2. **MigrationGuide.md** - What changed, breaking changes, deprecation list

## 📝 DOCUMENT TEMPLATES

---

# Template 1: SystemArchitecture.md

```markdown
# RumiAI System Architecture

## 🚀 Quick Start for New Developers
If you're new to this codebase, this document + a sample temporal windows JSON
(`/insights/[video_id]_temporal_windows_updated.json`) provides everything you need to understand the system.

### Key Entry Points
- **Start here**: `scripts/rumiai_runner.py` - Main entry point
- **Core processing**: `processors/temporal_compute.py` - Temporal window computation
- **Output examples**: `/insights/*/temporal_windows_updated.json` - See what we produce

## 🎯 System Overview

### Purpose
End-to-end video analysis pipeline that extracts ML features for creator performance prediction.

### High-Level Flow
```
Video URL → Scraping → ML Services → Timeline → Temporal Compute → Features → Insights
```

## 🏗️ Complete System Architecture

### System Flow Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  TikTok URL → Apify Scraper → Video File + Metadata             │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ML SERVICES LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │Vision Services│  │Audio Services│  │Analysis Svcs │         │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│  │ • YOLO       │  │ • Whisper    │  │ • FEAT       │         │
│  │ • MediaPipe  │  │ • Audio      │  │ • DeepFace   │         │
│  │ • OCR        │  │   Energy     │  │              │         │
│  │ • Scene Det  │  │              │  │              │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼─────────────────┼─────────────────┼──────────────────┘
          └─────────────────┼─────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TIMELINE BUILDER                              │
├─────────────────────────────────────────────────────────────────┤
│  Unified Timeline Creation                                       │
│  • Chronological ordering                                        │
│  • Entry type standardization                                    │
│  • Timestamp alignment                                          │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   TEMPORAL COMPUTE                               │
├─────────────────────────────────────────────────────────────────┤
│  Window Extraction → Feature Calculation → Metrics              │
│  • Hook (0-3s)                                                  │
│  • Middle Segments (~7.6s each)                                 │
│  • Closing (last 3s)                                            │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  Temporal Windows JSON → Insights → ML Models                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Service Dependencies Matrix

### Critical Processing Path
```
Video → MediaPipe → Timeline Builder → Temporal Compute → JSON Output
         ↓
    (All other services feed into Timeline Builder in parallel)
```

### Inter-Service Dependencies
| Service | Depends On | Required By | Can Run Standalone |
|---------|------------|-------------|-------------------|
| YOLO | UnifiedFrameManager | Creative Density | ✅ Yes |
| MediaPipe | UnifiedFrameManager | Framing, Gestures, Gaze | ✅ Yes |
| OCR | UnifiedFrameManager | Text Overlays | ✅ Yes |
| Scene Detection | Video file | Scene Pacing | ✅ Yes |
| Whisper | Audio extraction | Speech Features | ✅ Yes |
| Audio Energy | Audio extraction | Energy Patterns | ✅ Yes |
| FEAT | Video frames | Emotion Features | ✅ Yes |
| DeepFace | Video frames | Gender Detection | ✅ Yes |

### Shared Resources
```
UnifiedFrameManager (Shared by vision services)
├── Frame extraction at specified FPS
├── Frame caching for reuse
└── Memory management

AudioExtractor (Shared by audio services)
├── WAV extraction from video
├── Audio file caching
└── Cleanup management
```

## 🔄 Data Flow Specifications

### 1. Input Processing
```python
# Entry point: rumiai_runner.py
URL → ApifyClient.scrape_video()
    → Returns: video file + metadata JSON
```

### 2. ML Service Orchestration
```python
# Parallel processing: video_analyzer.py
async def analyze_video():
    tasks = [
        run_yolo(),
        run_whisper(),
        run_mediapipe(),
        run_ocr(),
        run_scene_detection(),
        run_audio_energy(),
        run_emotion_detection(),
        run_deepface()
    ]
    results = await asyncio.gather(*tasks)
```

### 3. Timeline Building
```python
# Unification: timeline_builder.py
def build_timeline(video_id, metadata, ml_results):
    timeline = Timeline(duration)
    timeline.add_yolo_entries(ml_results['yolo'])
    timeline.add_whisper_entries(ml_results['whisper'])
    timeline.add_mediapipe_entries(ml_results['mediapipe'])
    # ... etc
    return unified_analysis
```

### 4. Temporal Computation
```python
# Feature extraction: temporal_compute.py
def compute_temporal_windows(unified_analysis):
    hook = process_segment(0, 3, timelines)
    middle = process_middle_segments(timelines)
    closing = process_segment(end-3, end, timelines)
    return temporal_windows
```

### 5. Output Structure Example
```json
// Actual temporal window output:
{
  "video_id": "7515687288257465630",
  "duration": 44.0,
  "temporal_windows": {
    "hook": {
      "element_count": 29,
      "avg_density": 9.67,
      "word_count": 11,
      "eye_contact_rate": 0.67,
      "close_ratio": 0.0,
      "medium_ratio": 1.0,
      // ... 40+ more features
    },
    "middle_segments": [
      // 3-7 segments with same features
    ],
    "closing": {
      // Same features as hook
    }
  },
  "metadata": {
    "digg_count": 10100,
    "hashtag_analysis": {...}
  }
}
```

## 📁 File System Structure
```
/rumiai_v2
├── /api                    # Service implementations
│   ├── ml_services_unified.py
│   ├── apify_client.py
│   └── ...
├── /processors            # Core processing
│   ├── timeline_builder.py
│   ├── temporal_compute.py
│   └── video_analyzer.py
├── /ml_services          # ML service specifics
│   ├── audio_energy_service.py
│   ├── emotion_detection_service.py
│   └── ...
├── /config               # Configuration
└── /scripts             # Entry points
    └── rumiai_runner.py
```

## ⚡ Performance Characteristics

### Bottlenecks (from Phase 1 findings)
1. **OCR**: Slowest service (3.5s for 60s video)
2. **Frame extraction**: Redundant extraction across services
3. **Memory**: FEAT model uses 2GB+ RAM

### Optimization Status
| Component | Current | Optimized | Potential |
|-----------|---------|-----------|-----------|
| Total Pipeline | ~65s | ~45s | ~30s |
| Memory Peak | 4.5GB | 3.2GB | 2.5GB |
| GPU Utilization | 40% | 65% | 85% |

## 🔐 System Boundaries

### Input Requirements
- Video formats: MP4, MOV
- Max duration: 3 minutes
- Min resolution: 360p
- Audio: Required for speech/audio features

### Output Guarantees
- All temporal windows present
- Features normalized to 0-1 range
- Missing data handled gracefully
- JSON schema validated

## 🚀 Scaling Considerations

### Current Limitations
- Single video processing only
- No caching between videos
- Memory not released between services

### Future Architecture
- Batch processing capability
- Service containerization
- Queue-based architecture
- Distributed processing

---
```

# Template 2: MigrationGuide.md (For Legacy Users Only)

```markdown
# Migration Guide: Precompute → Temporal Architecture

Note: This guide is only needed if you're migrating from the old precompute system.
New developers should read SystemArchitecture.md instead.

## 🚨 BREAKING CHANGES

### Completely Removed
```
❌ DELETED: All precompute functions
❌ DELETED: 6-block temporal structure
❌ DELETED: compute_metrics() and variants
❌ DELETED: Block-based analysis
```

### Replaced Concepts
| Old (Precompute) | New (Temporal) | Migration Path |
|------------------|-----------------|----------------|
| 6 blocks (opening, development, etc.) | 3 windows (hook, middle, closing) | Remap features |
| compute_creative_density() | element_count in temporal windows | Update references |
| personTimeline | face_timeline | Change key names |
| aggregate_metrics | Individual features in windows | Decompose |

## 📋 Deprecation Timeline

### Already Removed (as of Phase 1 completion)
- [x] precompute_functions.py
- [x] precompute_functions_full.py
- [x] precompute_professional.py
- [x] Block-based segmentation

### Pending Removal (Phase 3 completion)
- [ ] Legacy test files referencing precompute
- [ ] Old documentation (8 legacy .md files)
- [ ] Unused config parameters

### Will Remain (Modified)
- [x] timeline_builder.py (enhanced for temporal)
- [x] ML services (now self-contained)
- [x] Frame extraction (unified manager)

## 🔄 Feature Mapping

### Feature Availability Changes
| Feature | Precompute | Temporal | Status | Notes |
|---------|------------|----------|--------|--------|
| Person framing | ✅ | ✅ | MODIFIED | Now includes average_face_size |
| Creative density | ✅ | ✅ | RENAMED | Now element_count, avg_density |
| Speech rate | ✅ | ✅ | UNCHANGED | word_count per window |
| Gesture count | ✅ | ✅ | IMPROVED | Better accuracy with 3 FPS |
| Hashtag analysis | ❌ | ✅ | FIXED | Was orphaned, now working |
| Emotion distribution | ✅ | ✅ | UNCHANGED | Same metrics |
| Scene changes | ✅ | ✅ | IMPROVED | Better scene boundaries |

### Features No Longer Available
| Feature | Reason | Alternative |
|---------|--------|-------------|
| Block energy distribution | 6-block structure removed | Use window energy metrics |
| Transition metrics | No block transitions | Use segment changes |
| Climax detection | Concept deprecated | Use energy burst patterns |

## 🛠️ Migration Instructions

### For Existing Code

#### Step 1: Update Imports
```python
# OLD - Remove these
from rumiai.precompute import compute_metrics
from rumiai.precompute.functions import analyze_video

# NEW - Use these
from rumiai_v2.processors.temporal_compute import compute_temporal_windows
from rumiai_v2.processors.video_analyzer import VideoAnalyzer
```

#### Step 2: Update Feature Access
```python
# OLD - Block-based access
hook_density = results['blocks']['opening']['creative_density']

# NEW - Window-based access
hook_density = results['temporal_windows']['hook']['element_count']
```

#### Step 3: Handle Renamed Features
```python
# Mapping dictionary for migration
FEATURE_MAP = {
    'creative_density': 'element_count',
    'person_timeline': 'face_timeline',
    'speech_rate': 'word_count',
    'camera_distance': 'framing_ratios'
}
```

### For ML Models

#### Feature Vector Changes
```python
# OLD feature vector (150+ features across 6 blocks)
old_features = flatten_blocks(six_blocks)

# NEW feature vector (~50 features × 3-7 windows)
new_features = flatten_temporal_windows(temporal_windows)
```

#### Retraining Requirements
1. Feature names have changed
2. Feature ranges may differ
3. Number of windows is variable (3-7 instead of fixed 6)
4. Some features discontinued

## 📊 Validation Procedures

### Output Structure Validation
```python
# Ensure temporal windows structure
assert 'temporal_windows' in output
assert 'hook' in output['temporal_windows']
assert 'middle_segments' in output['temporal_windows']
assert 'closing' in output['temporal_windows']

# Verify feature presence
required_features = [
    'element_count', 'word_count', 'gesture_count',
    'close_ratio', 'medium_ratio', 'wide_ratio'
]
for feature in required_features:
    assert feature in output['temporal_windows']['hook']
```

### Performance Comparison
| Metric | Precompute | Temporal | Improvement |
|--------|------------|----------|-------------|
| Processing time (60s video) | 85s | 65s | 23% faster |
| Memory usage | 5.2GB | 4.5GB | 13% less |
| Feature count | 150+ | ~350 | More granular |
| Accuracy | Baseline | +5-10% | Better windows |

## 🚧 Known Issues

### Temporary Issues (Being Fixed)
1. **Frame extraction redundancy** - Multiple services extract same frames
2. **Memory not released** - Services hold memory after completion
3. **No progress callbacks** - Can't track processing progress

### Won't Fix (By Design)
1. **No 6-block support** - Temporal windows are superior
2. **No backward compatibility** - Clean break from legacy
3. **No feature aggregation** - Models should handle windows directly

## 📚 Additional Resources

### Documentation
- Phase 1: Service Documentation (/services/*.md)
- Phase 2: Feature Documentation (/features/*.md)
- Phase 3: This guide + SystemArchitecture.md

### Test Files
- test_temporal_compute_v2.py - Main integration test
- benchmark_services.py - Performance testing
- validate_output_structure.py - Schema validation

### Support
- GitHub Issues: Report migration problems
- Slack: #rumiai-migration
- Wiki: Internal documentation

---

## ✅ Migration Checklist

Before considering migration complete:

### Code Migration
- [ ] All precompute imports removed
- [ ] Feature access updated to temporal windows
- [ ] Tests updated for new structure
- [ ] ML models retrained with new features

### Documentation
- [ ] All 11 new documents created
- [ ] Legacy documents archived
- [ ] README updated
- [ ] API documentation current

### Validation
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Output schema validated
- [ ] ML model accuracy verified

---

**MIGRATION STATUS**: In Progress (Phase 3 - Final Documentation)
**ESTIMATED COMPLETION**: [User to specify]
**BREAKING CHANGE DATE**: Precompute functions deleted on [Date]
```

## 📊 VALIDATION FOR PHASE 3

### Cross-Document Consistency
- [ ] Service names consistent across all phases
- [ ] Feature names match temporal_compute output
- [ ] File paths verified and accurate
- [ ] No contradictions between documents

### Completeness Check
- [ ] All services from Phase 1 included in architecture
- [ ] All features from Phase 2 mapped in migration guide
- [ ] All dependencies documented
- [ ] All breaking changes listed

### Migration Testing
- [ ] Test code runs without precompute
- [ ] Output structure validated
- [ ] Performance metrics accurate
- [ ] Known issues documented

## 🤝 USER CONSULTATION POINTS

During Phase 3, consult user for:
1. **Migration timeline** - When to delete precompute?
2. **Breaking changes** - Acceptable to break backward compatibility?
3. **Priority issues** - Which problems to fix first?
4. **Documentation gaps** - Anything missing from Phases 1 & 2?
5. **System boundaries** - Any constraints not documented?

---

**REMEMBER**:
- Phase 3 is SYNTHESIS - no new technical details
- Connect everything from Phases 1 & 2
- Focus on the complete picture
- Document what changed and why