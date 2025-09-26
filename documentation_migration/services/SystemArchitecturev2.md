# RumiAI System Architecture

  ## 📖 How to Use This Document
  - **This document**: System-wide architecture and data flow
  - **For ML service details**: See `/documentation_migration/services/*.md`
  - **For feature details**: See `/documentation_migration/services/*Features.md`
  - **For examples**: Check `/insights/*_temporal_windows_updated.json`

  ## 🏗️ System Overview

  ### Architecture Principles
  1. **Flexible Execution**: Services can run in parallel OR sequential mode
  2. **Fail-Fast**: Services validate before processing
  3. **Self-Contained**: Each service can run independently
  4. **Temporal Focus**: Features computed over time windows

  ### High-Level Data Flow
  ```
  TikTok URL → Metadata Scraping → Video Download → ML Services → Timeline → Temporal → JSON
                (Apify first)       (After metadata)    ↓
                                                   [Sequential or Parallel]
  ```

  ### Complete System Flow Diagram
  ```
  ┌─────────────────────────────────────────────────────────────────┐
  │                         INPUT LAYER                              │
  ├─────────────────────────────────────────────────────────────────┤
  │  TikTok URL → Apify Scraper → Metadata + Video File             │
  └────────────────────────┬────────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │                    ML SERVICES LAYER                             │
  │              (Sequential Mode - Default Order)                   │
  ├─────────────────────────────────────────────────────────────────┤
  │  1. YOLO (4.0s)           5. Scene Detection (3.0s)             │
  │  2. Whisper (26.1s)       6. Audio Energy (5.0s)                │
  │  3. MediaPipe (7.0s)      7. FEAT Emotion (74.0s) 🔴            │
  │  4. OCR (17.1s)           8. DeepFace Gender (6.0s)             │
  └────────────────────────┬────────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │                    TIMELINE BUILDER                              │
  ├─────────────────────────────────────────────────────────────────┤
  │  Unified Timeline Creation (timeline_builder.py)                │
  │  • Merges all ML outputs chronologically                        │
  │  • Standardizes entry formats                                   │
  │  • Resolves timestamp conflicts                                 │
  └────────────────────────┬────────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │                   TEMPORAL COMPUTE                               │
  ├─────────────────────────────────────────────────────────────────┤
  │  Window Extraction (temporal_compute.py)                        │
  │  ┌────────────┬──────────────────┬──────────────┐              │
  │  │ Hook       │ Middle Segments  │ Closing      │              │
  │  │ (0-3s)     │ (3-5 segments)   │ (last 3s)    │              │
  │  │ 50+ feats  │ 50+ per segment  │ 50+ feats    │              │
  │  └────────────┴──────────────────┴──────────────┘              │
  └────────────────────────┬────────────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │                      OUTPUT LAYER                                │
  ├─────────────────────────────────────────────────────────────────┤
  │  JSON Output: /insights/{video_id}_temporal_windows_updated.json │
  │  • 350+ features across temporal windows                        │
  │  • Engagement metadata                                          │
  │  • Processing metrics                                           │
  └─────────────────────────────────────────────────────────────────┘
  ```

  ## 📦 Service Layer

  ### Service Execution Model
  **Flexible Architecture - Sequential (Default) or Parallel**:

  ```python
  # ACTUAL execution order from video_analyzer.py
  analyses = {
      'yolo': self._run_yolo,
      'whisper': self._run_whisper,
      'mediapipe': self._run_mediapipe,
      'ocr': self._run_ocr,
      'scene_detection': self._run_scene_detection,
      'audio_energy': self._run_audio_energy,
      'emotion_detection': self._run_emotion_detection,  # FEAT
      'deepface_gender': self._run_deepface_gender
  }

  # Sequential Mode (default): Runs in order shown above
  # Parallel Mode: All run concurrently (PARALLEL_MODE=true)

  Service Categories & Performance (120s video, Sequential Mode)

  | Category | Services | Measured Time | % of Pipeline | Documentation |
  |----------|----------|---------------|---------------|---------------|
  | **Vision** | YOLO, MediaPipe, OCR, Scene | ~31s total | ~18% | [📘 services/VisionServices.md] |
  | **Audio** | Whisper, Audio Energy | ~31s total | ~18% | [📘 services/AudioServices.md] |
  | **Analysis** | FEAT, DeepFace | ~80s total | ~47% | [📘 services/AnalysisServices.md] |
  | **Metadata** | Apify Scraper | Network I/O | Pre-ML | [📘 services/MetadataServices.md] |

  Individual Service Performance (from InstrumentationResults.md):

  | Service | Sequential Time | % of Total | Status |
  |---------|----------------|------------|---------|
  | emotion_detection (FEAT) | 73.96s | 43.4% | 🔴 **BOTTLENECK** |
  | whisper | 26.14s | 15.3% | 🟡 Secondary bottleneck |
  | ocr | 17.09s | 10.0% | 🟢 Normal |
  | mediapipe | 7.01s | 4.1% | 🟢 Normal |
  | deepface_gender | 6.01s | 3.5% | 🟢 Normal |
  | audio_energy | 5.01s | 2.9% | 🟢 Normal |
  | yolo | 4.00s | 2.3% | 🟢 Normal |
  | scene_detection | 3.01s | 1.8% | 🟢 Normal |

  Critical Performance Bottlenecks

  ⚠️ FEAT consumes 40-60% of total processing time
  See [services/AnalysisServices.md#FEAT-Performance] for optimization opportunities

  🔄 Data Flow Architecture

  Stage 1: Metadata Scraping (FIRST)

  ```python
  # rumiai_runner.py:243 - Metadata comes first
  video_metadata = await self.apify.scrape_video(video_url)
  ```

  Stage 2: Video Download (AFTER metadata)

  ```python
  # rumiai_runner.py:249 - Download using metadata
  video_path = await self.apify.download_video(
      video_metadata.download_url,
      video_metadata.video_id
  )
  ```

  Stage 3: ML Processing (Sequential by default)

  ```
  Sequential Order (validated from code):
  1. YOLO → 2. Whisper → 3. MediaPipe → 4. OCR →
  5. Scene Detection → 6. Audio Energy →
  7. Emotion Detection (FEAT) → 8. DeepFace Gender
  ```

  Resource Sharing:
  - UnifiedFrameManager: Shared by vision services [→ services/VisionServices.md]
  - SharedAudioExtractor: Shared by audio services [→ services/AudioServices.md]

  Stage 4: Data Flow Split (Timeline vs Direct ML Data)

  **⚠️ CRITICAL: Not all services go through Timeline Builder**

  ```
  ML Services Results
        ├── Timeline-Based Services → Timeline Builder → Temporal Events
        │   (YOLO, Whisper, MediaPipe, OCR, Scene Detection, FEAT)
        │
        └── Direct ML Data Services → ml_data dict → Video-level Metadata
            (Audio Energy, DeepFace Gender)
  ```

  **Timeline Services** (temporal events with timestamps):
  ```python
  # processors/timeline_builder.py - Only for temporal services
  timeline.add_entry(TimelineEntry(
      entry_type='detection',
      start_time=0.5,
      end_time=0.7,
      data={'object': 'person'}
  ))
  ```

  **Direct ML Data Services** (video-level metadata):
  ```python
  # Goes directly to ml_data, bypasses timeline
  ml_data['audio_energy'] = {'rms_frames': [...], 'burst_pattern': 'middle_peak'}
  ml_data['deepface_gender'] = {'gender': 'female', 'confidence': 0.95}
  ```

  Stage 5: Temporal Window Computation

  # processors/temporal_compute.py
  def compute_temporal_windows(unified_analysis):
      hook = process_segment(0, 3)          # First 3 seconds
      middle = process_middle_segments()     # 3-5 segments based on duration
      closing = process_segment(-3, end)     # Last 3 seconds

  ### Temporal Window Bucket Thresholds
  | Video Duration | Middle Segments | Structure | Middle Duration | Segment Duration | Variance | ML Bucket |
  |----------------|-----------------|-----------|-----------------|------------------|----------|-----------|
  | 0-3s | None (null) | Hook only | N/A | N/A | N/A | 1 |
  | 3-9s | None (null) | Hook + Closing | N/A | N/A | N/A | 2 |
  | 9-13s | 3 segments | Hook + 3 Middle + Closing | 3-7s | 1.0-2.33s each | 2.33x | 3 |
  | 13-18s | 3 segments | Hook + 3 Middle + Closing | 7-12s | 2.33-4.0s each | 1.72x | 4 |
  | 18-33s | 4 segments | Hook + 4 Middle + Closing | 12-27s | 3.0-6.75s each | 2.25x | 5 |
  | 33-60s | 5 segments | Hook + 5 Middle + Closing | 27-54s | 5.4-10.8s each | 2.0x | 6 |
  | 60-90s | 5 segments | Hook + 5 Middle + Closing | 54-84s | 10.8-16.8s each | 1.56x | 7 |
  | 90-120s | 5 segments | Hook + 5 Middle + Closing | 84-114s | 16.8-22.8s each | 1.36x | 8 |
  | >120s | 5 segments (capped) | Hook + 5 Middle + Closing | >114s | >22.8s each | Variable | Beyond |

  **Critical Changes**:
  - The boundary changed from 6s to 9s for "no middle segments"
  - Videos ≤9s return `middle_segments: null` (not empty array `[]`)
  - Maximum segments capped at 5 for videos >75s

  **ML Training Note**: While 9-18s videos all output 3 middle segments (same structure), the ML training pipeline will bucket them separately as 9-13s and 13-18s to handle variance. The production output remains identical, but ML models train on duration-specific subsets.

  Feature Extraction Details:

  | Window      | Features        | Importance                      | Documentation
                  |
  |-------------|-----------------|---------------------------------|--------------------------
  ----------------|
  | Hook (0-3s) | 50+ features    | Highest - viewer decision point |
  [features/VisualFeatures.md#Hook]        |
  | Middle      | 50+ per segment | Variable by content             |
  [features/AudioFeatures.md#Middle]       |
  | Closing     | 50+ features    | High - CTA moment               |
  [features/BehavioralFeatures.md#Closing] |

  📊 Feature Categories

  Visual Features (21 total)

  - Person Framing: Face prominence metrics [→ services/VisualFeatures.md]
  - Creative Density: Visual complexity [→ services/VisualFeatures.md]
  - Scene Pacing: Edit rhythm [→ services/VisualFeatures.md]

  Audio Features (11 total)

  - Speech Metrics: Coverage, word count [→ services/AudioFeatures.md]
  - Energy Patterns: RMS, burst detection [→ services/AudioFeatures.md]

  Behavioral Features (11 total)

  - Emotions: 7 emotion ratios from FEAT [→ services/BehavioralFeatures.md]
  - Gaze: Eye contact patterns [→ services/BehavioralFeatures.md]

  Engagement Metrics (18 total)

  - Virality: Views, likes, shares [→ services/EngagementAndMetadata.md]

  🗂️ Complete File System Structure

  ### Core Processing Code
  ```
  /rumiai_v2/
  ├── api/                              # Service implementations
  │   ├── ml_services_unified.py       # Vision/Audio services
  │   ├── apify_client.py              # TikTok video scraping
  │   ├── whisper_cpp_service.py       # Speech transcription
  │   ├── shared_audio_extractor.py    # Audio extraction utility
  │   └── ...
  ├── ml_services/                      # ML service wrappers
  │   ├── emotion_detection_service.py # FEAT emotion (Timeline)
  │   ├── audio_energy_service.py      # Energy analysis (ML Data)
  │   ├── deepface_gender_service_simple.py # Gender detection
  │   └── ...
  ├── processors/                        # Core processing pipeline
  │   ├── temporal_compute.py          # ★ Feature extraction (350+ features)
  │   ├── timeline_builder.py          # Timeline unification
  │   ├── video_analyzer.py            # Service orchestration
  │   └── unified_frame_manager.py     # Frame extraction & caching
  ├── config/                           # Configuration
  │   └── settings.py                  # Environment & API configs
  └── utils/                            # Utilities
      ├── logger.py                   # Logging setup
      ├── file_handler.py             # File operations
      └── validators.py               # Data validation
  ```

  ### Scripts & Entry Points
  ```
  /scripts/
  ├── rumiai_runner.py                 # ★ Main entry point
  ├── run_deepface_gender.py          # Subprocess for gender detection
  └── test_temporal_compute_v2.py     # Testing temporal computation
  ```

  ### Output Directories
  ```
  /insights/                            # ★ Final unified JSON outputs
  │   └── {video_id}_temporal_windows_updated.json
  /unified_analysis/                    # Intermediate ML results
  │   └── {video_id}.json              # Timeline + ml_data
  /temp/                                # Downloaded videos
  │   └── {video_id}.mp4
  /logs/                                # Processing logs
  │   └── rumiai_{date}.log

  # Service-specific outputs (debugging)
  /emotion_detection_outputs/           # FEAT results
  /audio_energy_outputs/                # Audio analysis results
  /human_analysis_outputs/              # MediaPipe results
  /gender_detection_outputs/            # DeepFace results
  ```

  ### Service Data Flow Patterns

  | Service | Data Flow | Output Location | Timeline? |
  |---------|-----------|-----------------|-----------|
  | YOLO | Timeline-based | timeline entries | ✅ Yes |
  | Whisper | Timeline-based | timeline entries | ✅ Yes |
  | MediaPipe | Timeline-based | timeline entries | ✅ Yes |
  | OCR | Timeline-based | timeline entries | ✅ Yes |
  | Scene Detection | Timeline-based | timeline entries | ✅ Yes |
  | FEAT | Timeline-based | timeline entries | ✅ Yes |
  | Audio Energy | Direct ML Data | ml_data dict | ❌ No |
  | DeepFace Gender | Direct ML Data | ml_data dict | ❌ No |

  **Key Insight**: Services producing frame-by-frame events use Timeline.
  Services producing video-level metadata use Direct ML Data.

  ### 📊 Data Transformation Examples (Real Data from Production)

  #### Example 1: Timeline Service Flow (Whisper → Speech Feature)
  ```
  1. Whisper Output (ML Service):
     {"text": "Two minute TikTok videos", "start": 0.0, "end": 1.36}

  2. Timeline Entry (timeline_builder.py):
     {
       "entry_type": "speech",
       "start": 0.0,
       "end": 1.36,
       "data": {"text": "Two minute TikTok videos", "confidence": 0}
     }

  3. Temporal Window Feature (temporal_compute.py):
     Hook: {"word_count": 4, "speech_coverage": 0.45, "has_greeting": false}
  ```

  #### Example 2: Direct ML Data Flow (Audio Energy → Energy Feature)
  ```
  1. Audio Energy Output (ML Service):
     {
       "rms_frames": [0.1, 0.15, 0.2, ...],  // 31 frames per second
       "energy_windows": {"0-5s": 0.25, "5-10s": 0.45},
       "overall_stats": {"mean": 0.3, "max": 0.9}
     }

  2. ML Data Storage (bypasses timeline):
     ml_data["audio_energy"] = {full output above}

  3. Temporal Window Feature (temporal_compute.py reads ml_data):
     Hook: {"energy_level": 0.25, "energy_max": 0.45, "burst_pattern": "start"}
  ```

  #### Example 3: Data Merge in Temporal Windows
  ```json
  // unified_analysis/{video_id}.json contains:
  {
    "timeline": {
      "entries": [
        {"type": "speech", "start": 0.0, "end": 1.36, "data": {...}},
        {"type": "object", "start": 0.5, "end": 0.6, "data": {"class": "person"}}
      ]
    },
    "ml_data": {
      "audio_energy": {"rms_frames": [...], "burst_pattern": "middle_peak"},
      "deepface_gender": {"gender": "female", "confidence": 0.95}
    }
  }

  // temporal_compute.py merges both into:
  // insights/{video_id}_temporal_windows_updated.json
  {
    "temporal_windows": {
      "hook": {
        "word_count": 4,          // From timeline entries
        "person_count": 1,        // From timeline entries
        "energy_level": 0.25,     // From ml_data.audio_energy
        "avg_pitch_normalized": 1.2  // Uses ml_data.deepface_gender for normalization
      }
    }
  }
  ```

  #### Key Data Files in Pipeline
  | Stage | File Location | Content |
  |-------|--------------|---------|
  | ML Service Outputs | `/unified_analysis/{video_id}.json` | Raw timeline + ml_data |
  | Temporal Processing | `/insights/{video_id}_temporal_windows_updated.json` | Final features |
  | Frame Cache | `/tmp/rumiai_frames_{video_id}/` | Extracted frames |
  | Audio Cache | `/tmp/tmp*audio*.wav` | Extracted audio |

  ### 🎯 Final Output: The Unified JSON

  **Location**: `/insights/{video_id}_temporal_windows_updated.json`

  #### Complete Output Structure
  ```json
  {
    "video_id": "7428596413707144481",
    "duration": 18.0,
    "processing_timestamp": 1758314683.336669,
    "version": "2.0.0",

    "temporal_windows": {
      "hook": {
        // 50+ features for first 3 seconds
        "word_count": 15,
        "person_count": 1,
        "energy_level": 0.45,
        "close_ratio": 0.8,
        "element_count": 25,
        "has_greeting": true,
        "joy_ratio": 0.6,
        "eye_contact_rate": 0.75,
        // ... 40+ more features
      },

      "middle_segments": [
        // 3-5 segments based on duration (see bucket thresholds table)
        // null for videos ≤9s
        {
          "start": 3.0,
          "end": 10.6,
          "duration": 7.6,
          // Same 50+ features as hook
        },
        // ... more segments
      ],

      "closing": {
        // 50+ features for last 3 seconds
        "has_speech_cta": true,
        "energy_max": 0.9,
        // ... same features as hook
      }
    },

    "metadata": {
      // TikTok engagement metrics
      "digg_count": 10500,
      "play_count": 45000,
      "share_count": 1200,
      "comment_count": 350,

      // Video metadata
      "author": "@creator_username",
      "create_time": "2024-01-15T10:30:00Z",
      "description": "Video caption with #hashtags",

      // ML-derived metadata
      "gender_detection": {
        "gender": "female",
        "confidence": 0.95
      },
      "hashtag_analysis": {
        "hashtag_count": 5,
        "trending_count": 2,
        "niche_hashtags": ["#productreview", "#tutorial"]
      }
    }
  }
  ```

  #### Feature Categories in Each Window
  - **Visual**: 21 features (framing, density, scenes, objects)
  - **Audio**: 11 features (speech, energy, pitch)
  - **Behavioral**: 11 features (emotions, gaze, gestures)
  - **Total**: ~50+ features per temporal window

  #### Using the Output
  ```python
  # Load the unified JSON
  import json
  with open('insights/VIDEO_ID_temporal_windows_updated.json') as f:
      data = json.load(f)

  # Access temporal features
  hook_features = data['temporal_windows']['hook']
  middle_segments = data['temporal_windows']['middle_segments']
  closing_features = data['temporal_windows']['closing']

  # Access metadata
  engagement = data['metadata']['digg_count']
  gender = data['metadata']['gender_detection']['gender']
  ```

  🚨 Performance Characteristics & Optimization

  ### Actual Pipeline Performance (from InstrumentationResults.md)
  | Video Duration | Total Time | Processing Speed | Mode |
  |---------------|------------|------------------|------|
  | 18s | 83.96s | 0.21x realtime | Sequential |
  | 73s | 123.83s | 0.59x realtime | Sequential |
  | 120s | 177.52s | 0.68x realtime | Sequential |

  ### Thread Optimization Results
  | Service | Optimal Threads | Speedup | Environment Variable |
  |---------|----------------|---------|---------------------|
  | Whisper | 4 threads | 1.15x | `WHISPER_THREADS=4` |
  | OCR | 2 threads | 1.26x | `OMP_NUM_THREADS=2` |
  | YOLO | 2 threads | 1.05x | `CV2_THREADS=2` |

  ### Immediate Optimization Opportunities
  | Issue | Impact | Solution | Priority |
  |-------|--------|----------|----------|
  | FEAT processing time | 73.96s (43% of total) | Enable GPU acceleration | 🔴 HIGH |
  | Whisper performance | 26.14s (15% of total) | Use faster-whisper with GPU | 🟡 MEDIUM |
  | Frame extraction redundancy | ~10-15% overhead | Implement unified frame cache | 🟡 MEDIUM |
  | Memory not released | 2.5GB peak usage | Add garbage collection | 🟢 LOW |

  Architectural Debt

  1. Scene Detection not migrated to unified architecture
     → See services/VisionServices.md for current implementation
  2. No batch processing - one video at a time
     → Future enhancement planned

  🔗 Quick Reference Links

  ### For Understanding Services
  - **Vision Services**: `/services/VisionServices.md` (YOLO, MediaPipe, OCR, Scene)
  - **Audio Services**: `/services/AudioServices.md` (Whisper, Audio Energy)
  - **Analysis Services**: `/services/AnalysisServices.md` (FEAT, DeepFace)
  - **Performance Data**: `/services/ServicesPerformance.md`

  ### For Understanding Features
  - **Visual Features**: `/services/VisualFeatures.md` (framing, density, overlays)
  - **Audio Features**: `/services/AudioFeatures.md` (speech, energy, pitch)
  - **All Features List**: `/services/TotalFeatures.md` (complete 61 feature matrix)

  ### For Implementation Help
  - **Adding New Services**: `/NewServices.md` (integration patterns & lessons)
  - **Common Commands**: `/QUICK_REFERENCE.md#common-commands`
  - **Instrumentation Results**: `/InstrumentationResults.md` (performance data)

