# Stage 2: Video Processing - Implementation Summary

**Implementation Date**: 2025-10-13
**Source Specification**: VideoProcessingTI.md
**Status**: ✅ Complete - Ready for Integration Testing

---

## Overview

Stage 2 implements batch video processing with checkpoint/resume capability for the RumiAI ML Pipeline. It processes videos sequentially through the existing `scripts/rumiai_runner.py` production script with automatic recovery on interruption.

## What Was Implemented

### Core Modules Created

```
/home/jorge/rumiaifinal/ml_pipeline/stage2_processing/
├── __init__.py              # Package exports
├── main.py                  # Main orchestration (stage_2_video_processing_main)
├── exceptions.py            # Custom exceptions (4 types)
├── utils.py                 # Utility functions (save_json, load_json, get_bucket_path)
├── checkpoint.py            # Checkpoint management with backup/recovery
├── bucket_init.py           # Bucket directory initialization (8 buckets × 15 subdirs)
├── video_download.py        # Video download with retry logic
├── video_processor.py       # RumiAI integration + error handling
├── validation.py            # Schema validation for temporal_windows
└── pause_handler.py         # Graceful pause (Ctrl+C) support
```

### Key Features

1. **Batch Processing**
   - Processes up to 300 videos per bucket sequentially
   - Integrates with existing `scripts/rumiai_runner.py` via subprocess
   - Skip-on-fail policy (logs failures, continues batch)

2. **Checkpoint/Resume System**
   - Auto-resume without `--resume` flag needed
   - Checkpoint saved after each video (max 1 video lost on crash)
   - Backup checkpoint (.backup.json) for corruption recovery
   - Config validation prevents resume with different parameters

3. **Video Download**
   - Downloads from Apify URLs with 3 retry attempts
   - Exponential backoff (2s, 4s, 8s)
   - Resume optimization (skips already-downloaded valid files)
   - Minimum file size validation (100KB)

4. **Error Handling**
   - Custom exceptions: `DownloadError`, `ProcessingError`, `ValidationError`, `CheckpointCorruptionError`
   - Skip-on-fail for individual videos
   - Fail-fast for critical errors (disk full, permissions)
   - Detailed error logging with context

5. **Graceful Pause**
   - Ctrl+C pauses after current video completes
   - Second Ctrl+C force quits
   - Status saved in checkpoint (can resume later)

6. **Validation**
   - Input validation (video list, config.json)
   - Output schema validation (temporal_windows structure)
   - Post-processing validation (file counts match checkpoint)

---

## Architecture Integration

### How It Works with Existing RumiAI

```
Stage 2 ML Pipeline (NEW)
├── Downloads video from Apify URL
├── For each video:
│   └── subprocess.run(['python3', 'scripts/rumiai_runner.py', video_path])
│       └── EXISTING RumiAI pipeline processes video (9 ML services)
│       └── Generates: temporal_windows_updated.json
├── Validates output exists and has correct schema
└── Checkpoints progress after each video
```

**Key Point**: Stage 2 does NOT modify existing RumiAI code. It wraps `rumiai_runner.py` in a batch processing layer.

---

## Data Flow

### Input (from Stage 1)
```json
{
  "config": {
    "client_id": "acme_corp",
    "analysis_type": "hashtag",
    "target": "nutrition",
    "video_count": 100,
    // ... other config fields
  },
  "video_list": [
    {
      "id": "7428596413707144481",
      "videoMeta": {"downloadAddr": "https://..."},
      "duration": 25,
      "playCount": 50000
    }
  ],
  "bucket_name": "18-33s"
}
```

### Output (to Stage 2.4 / Stage 3)
```
/data/clients/acme_corp/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/
├── videos/
│   └── 7428596413707144481.mp4
├── analysis/insights/
│   └── 7428596413707144481_temporal_windows_updated.json  ← Main output
├── checkpoints/
│   ├── stage_2_checkpoint.json
│   └── stage_2_checkpoint.backup.json
└── logs/
    └── processing_2025-10-13.log
```

---

## Usage Example

### From Python (Stage Orchestration)
```python
from ml_pipeline.stage2_processing import stage_2_video_processing_main
from foundation.config import load_config

# Load config and video list from Stage 1
config = load_config("/data/clients/acme_corp/hashtags/nutrition/top_contrastive/config.json")
video_list = load_stage1_output()  # From Stage 1

# Process videos for a specific bucket
result = stage_2_video_processing_main(
    config=config,
    video_list=video_list,
    bucket_name="18-33s",
    enable_pause_support=True  # Ctrl+C graceful pause
)

print(f"Completed: {result['completed']}/{result['total']}")
print(f"Failed: {result['failed']}")
print(f"Status: {result['status']}")
```

### Command Line (Future Enhancement)
```bash
# Run Stage 2 for a specific bucket
python -m ml_pipeline.stage2_processing \
  --config /data/clients/acme/hashtags/nutrition/top_contrastive/config.json \
  --bucket 18-33s \
  --video-list stage1_output.json
```

---

## Checkpoint Resume Example

### Scenario: Process Interrupted at Video 45/100

**First Run** (interrupted):
```bash
# Start processing 100 videos
→ Processing video 1/100: 7428596413707144481
→ Processing video 2/100: 7428596413707144482
...
→ Processing video 45/100: 7428596413707144525
[SSH connection drops - process killed]
```

**Resume** (automatic):
```bash
# Re-run same command (no --resume flag needed)
✓ Checkpoint detected: 45/100 videos completed (45%)
  Last updated: 2 hours ago
  Failed videos: 2 (logged, will not retry)

→ Auto-resuming from bucket_18-33s (45/100 completed)
→ Skipping 45 already-processed videos
→ Processing remaining 55 videos...

→ Processing video 46/100: 7428596413707144526
...
```

---

## Error Handling

### Skip-on-Fail Examples

**Download Failure** (after 3 attempts):
```
❌ Failed to download video 7428596413707144481 after 3 attempts: Connection timeout
⚠️  Video 7428596413707144481 marked as failed. Continuing batch processing.
→ Processing video 47/100: 7428596413707144527
```

**RumiAI Processing Timeout** (>300s):
```
❌ RumiAI processing exceeded 300s timeout for video 7428596413707144482
⚠️  Video 7428596413707144482 marked as failed. Continuing batch processing.
→ Processing video 48/100: 7428596413707144528
```

**Invalid Output Schema**:
```
❌ Schema validation failed for video 7428596413707144483: field 'temporal_windows.hook' expected dict, got NoneType
⚠️  Video 7428596413707144483 marked as failed. Continuing batch processing.
→ Processing video 49/100: 7428596413707144529
```

### Fail-Fast Examples

**Disk Full**:
```
🔴 CRITICAL ERROR: Disk full. Cannot continue processing.
[Process exits with error code 1]
```

**Config Mismatch on Resume**:
```
❌ Config mismatch detected. Cannot resume with different parameters:
  video_count: checkpoint=100, current=150
[Process exits with error code 1]
```

---

## Performance Characteristics

### Expected Processing Time

Based on existing RumiAI benchmarks (from InstrumentationResults.md):

| Video Duration | RumiAI Processing | Download | Total per Video | 100 Videos |
|----------------|-------------------|----------|-----------------|------------|
| 18s            | ~84s              | ~15s     | ~99s            | ~2.75 hours |
| 60s            | ~124s             | ~25s     | ~149s           | ~4.1 hours |
| 120s           | ~178s             | ~35s     | ~213s           | ~5.9 hours |

**Batch Estimate**: 300 videos (mixed durations) ≈ **8-10 hours**

### Bottlenecks

1. **FEAT Emotion Detection**: 43% of RumiAI processing time
2. **Whisper Transcription**: 15% of RumiAI processing time
3. **Sequential Processing**: No parallelism (by design for reliability)

---

## Testing Recommendations

### Unit Tests Needed

```python
# ml_pipeline/tests/test_stage2_processing.py

def test_checkpoint_initialization_new():
    """Test creating new checkpoint"""
    pass

def test_checkpoint_initialization_resume():
    """Test loading existing checkpoint"""
    pass

def test_checkpoint_config_mismatch():
    """Test ValueError raised on config mismatch"""
    pass

def test_video_download_success():
    """Test successful download"""
    pass

def test_video_download_retry():
    """Test retry logic with exponential backoff"""
    pass

def test_video_download_skip_existing():
    """Test skip download if valid file exists"""
    pass

def test_rumiai_integration():
    """Test subprocess call to rumiai_runner.py"""
    pass

def test_schema_validation_valid():
    """Test valid temporal_windows schema"""
    pass

def test_schema_validation_invalid():
    """Test invalid schema raises ValidationError"""
    pass

def test_error_handling_skip_on_fail():
    """Test skip-on-fail for individual video errors"""
    pass

def test_error_handling_fail_fast():
    """Test fail-fast for critical errors (disk full)"""
    pass
```

### Integration Tests Needed

```python
# ml_pipeline/tests/test_stage2_integration.py

def test_end_to_end_5_videos():
    """
    Test complete video processing flow with 5 real videos.

    Setup:
    - Use real video list from Stage 1 (5 videos, bucket 18-33s)
    - Mock Apify downloads (return test MP4 files)
    - Mock RumiAI subprocess (return test temporal_windows JSON)

    Verify:
    - All 5 temporal_windows files created
    - Checkpoint status = "completed"
    - Stage 2.4 validation can load outputs
    """
    pass

def test_checkpoint_resume_after_interruption():
    """
    Test auto-resume without --resume flag.

    Setup:
    - Process 10 videos, interrupt after 5
    - Restart without --resume flag

    Verify:
    - Auto-resumes from checkpoint
    - Only remaining 5 videos processed
    - No duplicates in completed_video_ids
    """
    pass

def test_graceful_pause_on_sigint():
    """
    Test pause on Ctrl+C.

    Setup:
    - Process 10 videos, send SIGINT after video 5 completes

    Verify:
    - Processing pauses gracefully (not mid-video)
    - Checkpoint status="paused" saved
    - Resume processes only remaining 5 videos
    """
    pass
```

---

## Next Steps

### Immediate Tasks

1. **Integration with Stage 1**
   - Modify `ml_pipeline/stage1_discovery/video_discovery.py` to call Stage 2
   - Pass video list and config to `stage_2_video_processing_main()`

2. **Update Main Pipeline Orchestrator**
   - Modify `rumiai_ml_batch.py` to orchestrate Stage 1 → Stage 2

3. **Testing**
   - Unit tests for all modules
   - Integration test with 5 real videos
   - Checkpoint resume test

4. **Documentation**
   - Update QUICK_REFERENCE.md with Stage 2 usage
   - Add troubleshooting guide

### Future Enhancements (from VideoProcessingCHILD.md Section 9)

- **Phase 2**: Parallel video downloads (download next while processing current)
- **Phase 3**: Batch checkpoint writes (every 5 videos instead of every video)
- **Phase 4**: GPU-accelerated FEAT (73s → 15s per video, 5x speedup)
- **Phase 5**: Retry failed videos (`--retry-failed` flag)

---

## Dependencies

### Python Packages (Already Installed)
- `requests` - Video download
- `subprocess` - RumiAI integration
- Standard library: `os`, `json`, `logging`, `pathlib`, `signal`, `time`, `shutil`, `datetime`

### External Services
- Apify download URLs (HTTP GET)
- Existing RumiAI pipeline (`scripts/rumiai_runner.py`)

### Foundation Modules (Already Implemented)
- `foundation.paths` - Path construction
- `foundation.schemas` - Pydantic validation
- `foundation.constants` - Bucket definitions

---

## File Locations

### Implementation Files
```
/home/jorge/rumiaifinal/ml_pipeline/stage2_processing/
├── __init__.py              (59 lines)
├── main.py                  (140 lines)
├── exceptions.py            (89 lines)
├── utils.py                 (95 lines)
├── checkpoint.py            (174 lines)
├── bucket_init.py           (74 lines)
├── video_download.py        (82 lines)
├── video_processor.py       (177 lines)
├── validation.py            (89 lines)
└── pause_handler.py         (111 lines)

Total: ~1,090 lines of production code
```

### Documentation
```
/home/jorge/rumiaifinal/
├── documentation_migration/FutureDevelopments/ChildDocs/
│   ├── VideoProcessingCHILD.md        (HLD - Design)
│   └── VideoProcessingTI.md           (TI - Implementation Spec)
└── STAGE2_IMPLEMENTATION_SUMMARY.md   (This file)
```

---

## Implementation Completeness

✅ **Completed** (from VideoProcessingTI.md):
- [x] Section 2: Stage Contract (StageInput, StageOutput)
- [x] Section 3: Data Schemas (all schemas)
- [x] Section 4: Algorithmic Specifications (all 6 functions + helpers)
- [x] Section 5: Validation Rules (input/output validation)
- [x] Section 6: Error Handling (4 custom exceptions + handlers)
- [x] Section 9: Configuration & Environment (setup_environment)

⏳ **Pending**:
- [ ] Section 13: Test Specifications (unit tests, integration tests)
- [ ] Integration with Stage 1 output
- [ ] Integration with Stage 2.4 validation
- [ ] CLI wrapper (optional enhancement)

---

## Contact & Support

**Implementation Source**: VideoProcessingTI.md (v1.1)
**Implementation Date**: 2025-10-13
**Implemented By**: Claude Code
**Review Status**: Pending integration testing

For questions or issues, reference:
- Technical specification: `documentation_migration/FutureDevelopments/ChildDocs/VideoProcessingTI.md`
- High-level design: `documentation_migration/FutureDevelopments/ChildDocs/VideoProcessingCHILD.md`
- This summary: `STAGE2_IMPLEMENTATION_SUMMARY.md`
