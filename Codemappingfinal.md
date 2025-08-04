# Code Mapping for RumiAI Final - Complete Dependency Table

## Table of Contents
1. [Main Entry Points](#main-entry-points)
2. [Core Python Modules](#core-python-modules)
3. [ML Service Scripts](#ml-service-scripts)
4. [Processing and Analysis Scripts](#processing-and-analysis-scripts)
5. [API Integration Scripts](#api-integration-scripts)
6. [Utility and Support Scripts](#utility-and-support-scripts)
7. [Configuration and Template Files](#configuration-and-template-files)
8. [Node.js Integration Scripts](#nodejs-integration-scripts)
9. [Test and Debug Scripts](#test-and-debug-scripts)

## Main Entry Points

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `rumiai_runner.py` | `scripts/` | Main runtime orchestrator for RumiAI. Executes the full pipeline from TikTok video input to final Claude + PDF output. | TikTok video URL (CLI arg); env vars (`USE_CLAUDE_SONNET`, `USE_ML_PRECOMPUTE`, `OUTPUT_FORMAT_VERSION`); config flags | `/insights/[video_id]/[v2 .txt files]`; `/analysis_results/[video_id]_FullCreativeAnalysis.pdf`; temp logs; intermediate outputs | 10–80 MB per run depending on video length, metadata density, and prompt richness | CLI directly (user runs: `python rumiai_runner.py [video_url]`) | Real-time, per-video; manually triggered or batched in CLI script | **High** — failure disables pipeline; full system outage | `video_analyzer.py`, `timeline_builder.py`, `ml_data_extractor.py`, `prompt_builder.py`, all ML services | YOLO, Whisper, MediaPipe, EasyOCR, Claude API (Sonnet), PyMuPDF, OpenCV | Controlled via `.env`; `USE_ML_PRECOMPUTE` toggles Claude v2 pipeline; highly sensitive to prompt structure and memory limits |
| `compatibility_wrapper.js` | `scripts/` | Node.js bridge to Python rumiai_runner.py. Allows JavaScript code to spawn Python processes. | Same as rumiai_runner.py but via Node.js child_process | JSON data through stdout/stdin; exit codes | Same as rumiai_runner.py | Node.js services, test scripts | Per video analysis from JS | **Medium** — Node/Python bridge issues | `rumiai_runner.py` | child_process, Node.js runtime | Critical for JS/Python integration |

## Core Python Modules

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `settings.py` | `rumiai_v2/config/` | Settings configuration class. Loads environment variables and provides defaults. | Environment variables, .env file | Settings object with API keys, feature flags | < 1KB | All services during initialization | Once per run | **High** — missing API keys fail | None | python-dotenv | Must have CLAUDE_API_KEY, APIFY_API_TOKEN |
| `apify_client.py` | `rumiai_v2/api/` | Apify API client for TikTok video scraping and downloading | TikTok URL, API token | Video metadata dict, downloaded video path | Video file: 5-100MB | `rumiai_runner.py` | Once per video | **High** — scraping failure blocks pipeline | Apify API | aiohttp, requests | Actor ID: GdWCkxBtKWOsKjdch |
| `claude_client.py` | `rumiai_v2/api/` | Claude API client wrapper. Handles model selection and pricing. | Prompt text, model selection | Claude response text | 1-10KB per response | `rumiai_runner.py` | 7 times per video (7 prompts) | **High** — API failures block analysis | Anthropic API | anthropic SDK | Models: haiku, sonnet |
| `ml_services.py` | `rumiai_v2/api/` | ML services wrapper for YOLO, Whisper, MediaPipe, OCR, scene detection | Video file path | ML analysis results dict | 1-50MB depending on video | `video_analyzer.py` | Once per ML type per video | **Medium** — ML failures provide empty data | Individual ML scripts | subprocess, ffmpeg | Many services return empty results |
| `video_analyzer.py` | `rumiai_v2/processors/` | ML analysis orchestration. Runs all ML services in parallel. | Video file path | UnifiedAnalysis object | 1-10MB | `rumiai_runner.py` | Once per video | **Medium** — handles ML failures gracefully | All ML services | asyncio, concurrent.futures | Saves to respective output dirs |
| `timeline_builder.py` | `rumiai_v2/processors/` | Combines ML outputs into unified timeline with entries | UnifiedAnalysis object | Timeline object with entries | 100KB-1MB | `rumiai_runner.py` | Once per video | **Low** — robust error handling | ML validators | None | Entry types: object, speech, gesture, etc |
| `temporal_markers.py` | `rumiai_v2/processors/` | Generates time-based markers for key video events | UnifiedAnalysis object | Temporal markers dict | 10-100KB | `rumiai_runner.py` | Once per video | **Low** — non-critical feature | None | None | Identifies patterns and highlights |
| `ml_data_extractor.py` | `rumiai_v2/processors/` | Extracts ML data specific to each prompt type | UnifiedAnalysis, prompt type | PromptContext object | 10-500KB | `rumiai_runner.py` | 7 times per video | **Medium** — wrong extraction breaks prompts | None | None | Critical for Claude prompt data |
| `prompt_builder.py` | `rumiai_v2/processors/` | Builds Claude prompts with ML data and templates | PromptContext, template | Formatted prompt string | 5-50KB | `rumiai_runner.py` | 7 times per video | **Medium** — malformed prompts fail | `prompt_manager.py` | None | Uses _v2.txt templates |
| `output_adapter.py` | `rumiai_v2/processors/` | Converts v2 format to v1 for backward compatibility | v2 format dict, prompt type | v1 format dict | Same size | `rumiai_runner.py` | When OUTPUT_FORMAT_VERSION=v1 | **Low** — format conversion only | None | None | Legacy support |
| `precompute_functions.py` | `rumiai_v2/processors/` | Wrapper functions for compute functions. Maps prompt types. | UnifiedAnalysis dict | Precomputed metrics dict | 10-100KB | `rumiai_runner.py` | 7 times per video | **High** — compute failures break prompts | `precompute_functions_full.py` | None | Contains COMPUTE_FUNCTIONS mapping |
| `precompute_functions_full.py` | `rumiai_v2/processors/` | Actual compute logic. Generates 30-50 metrics per prompt type. | Timeline data, duration | Metrics dict for prompt | 10-50KB | `precompute_functions.py` | 7 times per video | **High** — core metric generation | None | numpy, various Python libs | 3000+ lines of compute logic |
| `prompt_manager.py` | `rumiai_v2/prompts/` | Template management. Loads and formats prompt templates. | Template dir path | Formatted prompts | 5-20KB per prompt | `prompt_builder.py` | 7 times per video | **Medium** — missing templates fail | File system | None | Loads all _v2.txt at startup |
| `response_validator.py` | `rumiai_v2/validators/` | Validates Claude responses for 6-block structure | Response text, prompt type | Validation result, normalized data | Same as input | `rumiai_runner.py` | 7 times per video | **Medium** — validation failures need retry | None | json | Maps block names by prompt type |

## ML Service Scripts

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `whisper_transcribe.py` | `(MISSING)` | Speech transcription using OpenAI Whisper | Video/audio file path | Transcript with segments | 1-50KB | `ml_services.py` | Once per video | **High** — missing file | None | openai-whisper | Referenced but doesn't exist |
| `scene_detection.py` | `local_analysis/` | Scene detection using PySceneDetect | Video file path | Scene list with timestamps | 5-20KB | Used standalone | Once per video | **Low** — has implementation | None | scenedetect[opencv] | Working implementation exists |
| `enhanced_human_analyzer.py` | `local_analysis/` | MediaPipe human pose and gesture analysis | Video file path | Pose/gesture data | 10-100KB | Used standalone | Once per video | **Low** — complete implementation | None | mediapipe | Could replace empty MediaPipe service |
| `object_tracking.py` | `local_analysis/` | YOLOv8 object detection with DeepSort tracking | Video file path | Object detections with tracking | 50-500KB | Used standalone | Once per video | **Low** — working code | None | ultralytics, deep-sort-realtime | Could replace empty YOLO service |
| `scene_labeling.py` | `local_analysis/` | Scene classification and labeling | Scene data | Labeled scenes | 10-50KB | After scene detection | Once per video | **Low** — enhancement feature | `scene_detection.py` | None | Adds semantic labels to scenes |
| `content_moderation.py` | `local_analysis/` | Content safety and moderation checks | Video frames | Safety scores | 1-10KB | Optional | Once per video | **Low** — safety feature | None | Various ML models | Not integrated into main pipeline |
| `frame_sampler.py` | `local_analysis/` | Frame extraction utilities | Video file, sampling rate | Extracted frames | Variable | ML services | Multiple times | **Low** — utility function | None | opencv-python | Supports other ML services |
| `mediapipe_human_detector.py` | Root | Standalone MediaPipe implementation | Video file path | Human detection data | 10-100KB | Could be integrated | Once per video | **Low** — alternative implementation | None | mediapipe | Working MediaPipe code |
| `detect_tiktok_creative_elements.py` | Root | Comprehensive ML analysis combining multiple models | Video file path | Combined ML analysis | 100KB-1MB | Could replace ml_services.py | Once per video | **Low** — complete implementation | None | YOLO, EasyOCR, MediaPipe | Contains all ML implementations |

## Processing and Analysis Scripts

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `analysis.py` | `rumiai_v2/core/models/` | Data models for UnifiedAnalysis, MLAnalysisResult | Dict data | Model objects | N/A (models) | Throughout system | Constantly | **High** — core data structure | None | dataclasses | Central data model |
| `timeline.py` | `rumiai_v2/core/models/` | Timeline and TimelineEntry data models | Timeline data | Timeline objects | N/A (models) | `timeline_builder.py` | Per video | **High** — core data structure | None | dataclasses | Stores temporal data |
| `timestamp.py` | `rumiai_v2/core/models/` | Timestamp handling and conversion | Various timestamp formats | Timestamp objects | N/A (models) | Throughout system | Constantly | **Medium** — format issues | None | Standard library | Handles time formats |
| `prompt.py` | `rumiai_v2/core/models/` | PromptType enum, PromptBatch, PromptResult models | Prompt data | Model objects | N/A (models) | Prompt system | Per prompt | **Medium** — prompt structure | None | enum, dataclasses | Defines prompt types |
| `video.py` | `rumiai_v2/core/models/` | VideoMetadata model | Video metadata dict | VideoMetadata object | N/A (models) | Throughout system | Per video | **Low** — data container | None | dataclasses | Stores video info |
| `ml_data_validator.py` | `rumiai_v2/core/validators/` | Validates ML data formats and content | ML output data | Validated data, errors | Same as input | ML services | Per ML output | **Medium** — data quality | None | Standard library | Ensures data integrity |
| `timeline_validator.py` | `rumiai_v2/core/validators/` | Validates timeline entries and structure | Timeline data | Validation results | Small | `timeline_builder.py` | Per timeline operation | **Low** — data validation | None | Standard library | Ensures timeline consistency |
| `timestamp_validator.py` | `rumiai_v2/core/validators/` | Validates and normalizes timestamps | Timestamp strings/objects | Valid timestamps | Small | Throughout system | Constantly | **Low** — format validation | None | Standard library | Handles various formats |

## API Integration Scripts

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `__init__.py` | `rumiai_v2/api/` | API package initialization | None | Exports | N/A | Import statements | On import | **Low** — package init | All API modules | None | Makes imports cleaner |
| `rumiai.py` | `rumiai_v2/api/` | High-level RumiAI API interface | Video URL, options | Analysis results | Variable | Could be used | Per analysis | **Medium** — alternative interface | All services | None | Not used by runner |

## Utility and Support Scripts

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `file_handler.py` | `rumiai_v2/utils/` | File I/O utilities, JSON operations | File paths, data | Read/written files | Variable | Throughout system | Constantly | **Low** — has error handling | File system | json, pathlib | Central file operations |
| `logger.py` | `rumiai_v2/utils/` | Logging configuration and colored output | Log messages | Formatted logs | N/A (stdout) | All modules | Constantly | **Low** — logging only | None | colorama, logging | Colored console output |
| `metrics.py` | `rumiai_v2/utils/` | Performance metrics and video processing stats | Timing data, metrics | Metrics objects | Small | `rumiai_runner.py` | Throughout run | **Low** — monitoring only | None | time, psutil | Tracks performance |
| `exceptions.py` | `rumiai_v2/core/` | Custom exception classes | Error conditions | Exception objects | N/A | Throughout system | On errors | **Low** — error handling | None | Standard library | Defines custom errors |

## Configuration and Template Files

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `creative_density_v2.txt` | `prompt_templates/` | Claude prompt template for creative density analysis | Template variables | Formatted prompt | 5-10KB | `prompt_manager.py` | Once per video | **Medium** — prompt quality | None | None | Defines 6-block structure |
| `emotional_journey_v2.txt` | `prompt_templates/` | Claude prompt template for emotional journey | Template variables | Formatted prompt | 5-10KB | `prompt_manager.py` | Once per video | **Medium** — prompt quality | None | None | Emotion analysis prompt |
| `speech_analysis_v2.txt` | `prompt_templates/` | Claude prompt template for speech analysis | Template variables | Formatted prompt | 5-10KB | `prompt_manager.py` | Once per video | **Medium** — prompt quality | None | None | Speech patterns prompt |
| `visual_overlay_analysis_v2.txt` | `prompt_templates/` | Claude prompt template for visual overlays | Template variables | Formatted prompt | 5-10KB | `prompt_manager.py` | Once per video | **Medium** — prompt quality | None | None | Text/sticker analysis |
| `metadata_analysis_v2.txt` | `prompt_templates/` | Claude prompt template for metadata analysis | Template variables | Formatted prompt | 5-10KB | `prompt_manager.py` | Once per video | **Medium** — prompt quality | None | None | Metadata insights |
| `person_framing_v2.txt` | `prompt_templates/` | Claude prompt template for person framing | Template variables | Formatted prompt | 5-10KB | `prompt_manager.py` | Once per video | **Medium** — prompt quality | None | None | Human framing analysis |
| `scene_pacing_v2.txt` | `prompt_templates/` | Claude prompt template for scene pacing | Template variables | Formatted prompt | 5-10KB | `prompt_manager.py` | Once per video | **Medium** — prompt quality | None | None | Scene rhythm analysis |
| `.env.example` | Root | Environment variable template | None | Example config | < 1KB | User reference | Once | **Low** — documentation | None | None | Shows required vars |
| `temporal_markers.json` | `config/` | Temporal marker configuration | None | Config settings | < 5KB | `temporal_markers.py` | Once per run | **Low** — config file | None | None | Marker settings |

## Node.js Integration Scripts

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `LocalVideoAnalyzer.js` | `server/services/` | Node.js service to run ML analysis scripts | Video path | Analysis results | Variable | Node.js API | Per video | **Medium** — subprocess management | Python scripts | child_process | Spawns Python processes |
| `TemporalMarkerService.js` | `server/services/` | Node.js temporal marker generation | Analysis data | Markers | 10-100KB | Node.js API | Per video | **Low** — data processing | None | None | JS implementation |
| `WhisperTranscriptionService.js` | `server/services/` | Node.js Whisper transcription service | Audio file | Transcript | 1-50KB | Node.js API | Per video | **Medium** — external process | Whisper binary | child_process | Alternative to Python |
| `ClaudeService.js` | `server/services/` | Node.js Claude API integration | Prompts | Responses | Variable | Node.js API | Per prompt | **High** — API dependency | Claude API | axios | JS Claude client |
| `run_claude_prompts_with_delays.js` | Root | Batch prompt execution with delays | Video data | All prompts results | Variable | Manual/automation | Batch processing | **Medium** — orchestration | Claude API | None | Prevents rate limits |
| `test_rumiai_complete_flow.js` | Root | End-to-end testing script | Test data | Test results | Small | Testing | During tests | **Low** — testing only | All services | None | Integration tests |
| `resume_analysis.js` | Root | Resume interrupted analysis | Partial results | Complete analysis | Variable | After failures | As needed | **Low** — recovery tool | All services | None | Failure recovery |
| `UnifiedTimelineAssembler.js` | Root | Timeline assembly in JavaScript | ML outputs | Unified timeline | 100KB-1MB | Node.js pipeline | Per video | **Medium** — data assembly | None | None | Alternative to Python |

## Test and Debug Scripts

| **File Name** | **Directory** | **Description** | **Data In** | **Data Out** | **Output Size (est)** | **Called By** | **How Often** | **Risk** | **Dep. Services** | **Dep. 3rd Party** | **Notes** |
|---------------|---------------|-----------------|-------------|--------------|----------------------|---------------|---------------|----------|-------------------|-------------------|-----------|
| `test_scene_pacing_debug.py` | Root | Debug script for scene pacing issues | Test scene data | Debug output | Console output | Manual debugging | As needed | **None** — debug tool | `parse_timestamp_to_seconds` | None | Created to fix scene bug |
| `test_fps_fix.py` | Root | Test script for FPS-related fixes | Video data | Test results | Console output | Manual testing | As needed | **None** — test tool | Various | None | FPS handling tests |
| `test_e2e.py` | Root | End-to-end Python tests | Test videos | Test results | Test reports | pytest | During tests | **None** — testing | All services | pytest | Python integration tests |
| `setup.sh` | Root | Environment setup script | None | Installed environment | N/A | Manual setup | Once | **High** — initial setup | apt, pip | bash | Creates venv, installs deps |
| `safe_video_analysis.sh` | Root | Safe execution wrapper script | Video path | Analysis results | Variable | Manual/automation | Per video | **Low** — safety wrapper | Python scripts | bash | Adds error handling |
| `run_tests.sh` | Root | Test execution script | None | Test results | Test output | CI/CD | Per commit | **Low** — testing | pytest | bash | Runs test suite |

## Summary Statistics

- **Total Python Scripts**: 45+
- **Total Node.js Scripts**: 12+
- **Total Configuration Files**: 10+
- **Missing/Broken Scripts**: 4 (whisper_transcribe.py, empty ML implementations)
- **High Risk Components**: 8 (API clients, main runner, core models)
- **3rd Party Dependencies**: 40+ Python packages, 15+ Node.js packages

## Critical Path for rumiai_runner.py

1. **rumiai_runner.py** → Entry point
2. **settings.py** → Configuration
3. **apify_client.py** → Video download
4. **ml_services.py** → ML analysis (mostly empty)
5. **video_analyzer.py** → ML orchestration
6. **timeline_builder.py** → Data combination
7. **precompute_functions.py** → Metric calculation
8. **prompt_builder.py** → Claude prompts
9. **claude_client.py** → API calls
10. **response_validator.py** → Output validation

## Key Issues Identified

1. **ML Services Implementation**: Most ML services return empty results
2. **Missing whisper_transcribe.py**: Referenced but doesn't exist
3. **GPU Support**: Must install correct PyTorch version (CUDA vs CPU)
4. **Scene Data Integration**: Fixed bug where scene count was wrong
5. **Working ML Code**: Exists in standalone scripts but not integrated

This comprehensive mapping provides a complete view of all components in the rumiaifinal codebase, their relationships, and current status.