# Checkpoint Resume System

**Document Type**: High-Level Design (HLD)
**Technical Implementation**: See [MLCheckpointResumeTI.md](./MLCheckpointResumeTI.md) for code, integration details, and testing
**Last Updated**: 2025-10-01

---

## Overview

### Business Problem
Long-running batch analyses (6-8 hours for 300+ videos) are vulnerable to interruptions:
- **SSH/Terminal disconnections** (WiFi drops, VPN issues, laptop sleep)
- **System crashes** (out of memory, GPU driver failures, power outages)
- **Manual interruptions** (code fixes, parameter adjustments, user Ctrl+C)

Without checkpoint resume, interruptions waste hours of compute time and require re-processing hundreds of completed videos.

### Solution
Implement automatic checkpoint system that:
1. **Saves progress** after each video completes analysis
2. **Auto-resumes** on restart (detects checkpoint, skips completed videos)
3. **Tracks failures** separately (logs failed videos without stopping batch)

### Stakeholder Value
- **Tumi Labs**: Reliable batch processing reduces wasted compute costs and delays
- **Operations**: Can confidently run overnight analyses without babysitting
- **Development**: Enables iterative fixes mid-batch without losing progress

---

## System Design

### Checkpoint Strategy

**When checkpoints are created**: Automatically after each video completes RumiAI analysis

**When checkpoints are used**: Automatically when restarting same analysis (auto-resume)

**How to override**: Use `--force` flag to discard checkpoint and restart fresh

---

## Checkpoint File Structure

### Location
```
/data/clients/{client_id}/{analysis_type}/{target}/checkpoints/analysis_state.json
```

### Schema
```json
{
  "run_id": "20250128_143052",
  "client_id": "client_acme_corp",
  "analysis_type": "hashtag",
  "target": "#nutrition",
  "status": "in_progress",

  "config": {
    "video_count": 300,
    "date_filter": "last_90_days"
  },

  "timestamps": {
    "started_at": "2025-01-28T14:30:52Z",
    "last_updated": "2025-01-28T16:45:12Z"
  },

  "progress": {
    "total_videos_target": 480,
    "videos_completed": 127,
    "videos_failed": 3,
    "videos_remaining": 350,
    "percentage": 26.5
  },

  "bucket_progress": {
    "bucket_0-3s": {
      "status": "completed",
      "videos_processed": 60,
      "videos_failed": 0,
      "completed_video_ids": ["7428596413707144481", "7429..."]
    },
    "bucket_3-9s": {
      "status": "completed",
      "videos_processed": 60,
      "videos_failed": 0,
      "completed_video_ids": ["7430...", "7431..."]
    },
    "bucket_9-13s": {
      "status": "in_progress",
      "videos_processed": 7,
      "videos_failed": 0,
      "videos_target": 60,
      "completed_video_ids": ["7432...", "7433..."],
      "current_video": "7434596413707144999"
    },
    "bucket_13-18s": {
      "status": "not_started",
      "videos_processed": 0,
      "videos_failed": 0,
      "videos_target": 60
    }
  },

  "failed_videos": [
    {
      "video_id": "7435596413707145000",
      "bucket": "bucket_9-13s",
      "error": "Download failed: 404 Not Found",
      "timestamp": "2025-01-28T15:22:10Z",
      "retry_count": 0
    },
    {
      "video_id": "7436596413707145001",
      "bucket": "bucket_9-13s",
      "error": "FEAT timeout after 300s",
      "timestamp": "2025-01-28T16:10:33Z",
      "retry_count": 0
    }
  ]
}
```

---

## CLI Usage

### Starting Fresh Analysis

```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type hashtag \
  --target "#nutrition" \
  --video-count 300 \
  --date-filter "last_90_days"
```

**Behavior**:
- ✅ Check if checkpoint exists for this client/analysis-type/target
- ✅ If exists with status "in_progress" → Auto-resume (see below)
- ✅ If not exists → Create new checkpoint, start analysis
- ✅ If exists with status "completed" → Warn user, suggest `--force` to restart

---

### Auto-Resume (Default Behavior)

```bash
# Process was interrupted at video 127/480
# User simply re-runs the SAME command:

python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type hashtag \
  --target "#nutrition" \
  --video-count 300 \
  --date-filter "last_90_days"
```

**Output**:
```
✓ Checkpoint detected: 127/480 videos completed (26.5%)
  Last updated: 2 hours ago
  Failed videos: 3 (logged, will not retry)

→ Auto-resuming from bucket_9-13s (7/60 completed)
→ Skipping 127 already-processed videos
→ Processing remaining 353 videos...

Processing video 128/480 (bucket_9-13s: 8/60)...
```

**Behavior**:
- ✅ Loads checkpoint automatically
- ✅ Validates config matches (video_count, date_filter)
- ✅ Skips completed videos
- ✅ Continues from last position
- ✅ Failed videos remain logged but not retried (skip-on-fail policy)

---

### Force Restart (Discard Checkpoint)

```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type hashtag \
  --target "#nutrition" \
  --video-count 300 \
  --date-filter "last_90_days" \
  --force
```

**Output**:
```
⚠️  Checkpoint found: 127/480 videos completed (26.5%)
    This will discard all progress and restart from beginning.

Continue? (y/N): y

→ Checkpoint discarded
→ Starting fresh analysis...

Processing video 1/480 (bucket_0-3s: 1/60)...
```

**Behavior**:
- ✅ Prompts for confirmation (safety check)
- ✅ Backs up old checkpoint to `analysis_state_backup_20250128.json`
- ✅ Creates new checkpoint
- ✅ Starts analysis from beginning

---

## Implementation Logic

### Checkpoint Manager Class

**Key Responsibilities**:
- Load/create checkpoint files
- Mark videos as completed or failed
- Track progress across duration buckets
- Validate configuration on resume
- Provide resume point for interrupted batches

**Core Methods**:
- `save_config()` - Initialize checkpoint with configuration
- `validate_config()` - Ensure config matches on resume
- `save_progress()` - Mark video completed, update checkpoint
- `get_resume_point()` - Get position to resume from
- `load_completed_features()` - Load all features for ML training
- `clear_checkpoint()` - Force restart (discard progress)

**Implementation**: See [MLCheckpointResumeTI.md - CheckpointManager Implementation](./MLCheckpointResumeTI.md#checkpointmanager-implementation)

---

### Batch Processing with Auto-Resume

**Workflow**:
1. Check for existing checkpoint
2. If exists: Validate config, load completed IDs, auto-resume
3. If not exists (or --force): Create new checkpoint, start fresh
4. Fetch videos via Apify
5. Bucket and select videos
6. Filter out completed videos
7. Process remaining videos with checkpoint updates
8. Mark analysis complete

**Key Behaviors**:
- **Auto-resume**: Automatically detects and resumes from checkpoint (no flag needed)
- **Config validation**: Prevents resuming with different parameters
- **Skip completed**: Filters out already-processed videos
- **Fail-fast**: Errors stop batch for debugging (not skip-on-fail)

**Implementation**: See [MLCheckpointResumeTI.md - Integration Example](./MLCheckpointResumeTI.md#integration-example)

---

## Edge Cases & Handling

### Case 1: Config Mismatch During Resume

**Scenario**: User changes parameters when resuming

```bash
# Original command
python rumiai_ml_batch.py --client "acme" --target "#nutrition" --video-count 300

# User tries different video-count
python rumiai_ml_batch.py --client "acme" --target "#nutrition" --video-count 250
```

**Handling**:
```
✗ Checkpoint config mismatch:
  - Checkpoint video_count: 300
  - Provided video_count: 250

Options:
  1. Match original config (remove --video-count 250)
  2. Use --force to discard checkpoint and restart
```

---

### Case 2: Checkpoint File Corrupted

**Scenario**: JSON parse error when loading checkpoint

```bash
python rumiai_ml_batch.py --client "acme" --target "#nutrition"
```

**Handling**:
```
✗ Checkpoint file corrupted (JSON parse error)

Options:
  1. Restore from backup: analysis_state_backup_20250128.json
  2. Use --force to create new checkpoint
```

**Implementation**: Checkpoint manager detects corruption, suggests backup restore or --force flag to create fresh checkpoint

---

### Case 3: Multiple Analyses for Same Target

**Scenario**: User wants to run different configs for same hashtag

```bash
# First analysis: 300 videos, last 90 days
python rumiai_ml_batch.py --client "acme" --target "#nutrition" --video-count 300 --date-filter last_90_days

# Later: wants 250 videos, last 30 days (different analysis)
python rumiai_ml_batch.py --client "acme" --target "#nutrition" --video-count 250 --date-filter last_30_days
```

**Handling**: Different configs = different analyses, need separate checkpoint paths

**Solution**: Include config hash in checkpoint filename
```
checkpoints/
  ├── analysis_state_config_abc123.json  # 300 videos, 90 days
  └── analysis_state_config_def456.json  # 250 videos, 30 days
```

---

### Case 4: Apify Videos No Longer Available

**Scenario**: Some videos deleted from TikTok between interruption and resume

**Handling**:
- Mark as failed with reason "Video no longer available"
- Continue with remaining videos
- Include in final summary

---

### Case 5: Disk Full Mid-Batch

**Scenario**: Disk fills up, can't write checkpoint or video files

```bash
Processing video 127/480...
ERROR: [Errno 28] No space left on device
```

**Handling**: Stop batch immediately, preserve last saved checkpoint, prompt user to free disk space and resume

---

## Testing Strategy

### Testing Strategy

**Unit Tests Required**:
- ✅ Checkpoint creation and initialization
- ✅ Config save and validation
- ✅ Config mismatch detection
- ✅ Save/load progress
- ✅ Resume point calculation
- ✅ Load completed features
- ✅ Clear checkpoint (force restart)

**Integration Tests Required**:
- ✅ Full batch with simulated interruption
- ✅ Auto-resume verification (no duplicates)
- ✅ Config mismatch error handling
- ✅ Checkpoint corruption recovery

**Implementation**: See [MLCheckpointResumeTI.md - Testing](./MLCheckpointResumeTI.md#testing)

---

## Performance Considerations

### Checkpoint Write Frequency

**Current**: After each video (480 writes for full batch)

**Optimization**: Batch writes every N videos (e.g., every 5 videos)
- **Tradeoff**: Lose up to 5 videos of progress if interrupted mid-batch
- **Benefit**: Reduce disk I/O by 80%

**Recommendation**: Start with per-video writes (simple, safe), optimize later if needed

---

### Checkpoint File Size

**Typical size**: ~50KB for 480 videos (JSON with video IDs)

**Growth**: Linear with video count (not a concern)

---

## Future Enhancements

### Retry Logic for Failed Videos
Currently: Skip-on-fail (no retries)

Future: Add `--retry-failed` flag to re-process failed videos from previous run

```bash
# After batch completes with 5 failed videos
python rumiai_ml_batch.py --client "acme" --target "#nutrition" --retry-failed
```

---

### Progress Dashboard
Real-time web dashboard showing:
- Current progress percentage
- Estimated time remaining
- Failed videos list
- Processing speed (videos/hour)

---

### Distributed Processing
Split buckets across multiple machines, merge checkpoints

```bash
# Machine 1: Process buckets 0-3
python rumiai_ml_batch.py --client "acme" --target "#nutrition" --buckets 0-3

# Machine 2: Process buckets 4-7
python rumiai_ml_batch.py --client "acme" --target "#nutrition" --buckets 4-7
```

---

## Summary

### Key Decisions
- ✅ **Auto-resume**: Default behavior (no `--resume` flag needed)
- ✅ **Skip-on-fail**: Failed videos logged but don't stop batch
- ✅ **Per-video checkpoints**: Save after each video completes
- ✅ **Force restart**: `--force` flag to discard checkpoint
- ✅ **Config validation**: Prevent accidental resume with different parameters

### Success Metrics
- **Time saved**: 3-6 hours per interruption
- **Reliability**: 100% recovery from interruptions
- **Usability**: Zero extra flags for normal operation (auto-resume "just works")