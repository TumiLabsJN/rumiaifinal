# Checkpoint Resume System

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

```python
class CheckpointManager:
    def __init__(self, checkpoint_path):
        self.path = checkpoint_path
        self.data = self._load_or_create()

    def _load_or_create(self):
        """Load existing checkpoint or create new one"""
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                return json.load(f)
        return self._create_new_checkpoint()

    def mark_video_completed(self, video_id, bucket):
        """Mark video as completed and save checkpoint"""
        # Update bucket progress
        self.data['bucket_progress'][bucket]['videos_processed'] += 1
        self.data['bucket_progress'][bucket]['completed_video_ids'].append(video_id)

        # Update overall progress
        self.data['progress']['videos_completed'] += 1
        self.data['progress']['videos_remaining'] -= 1
        self.data['progress']['percentage'] = (
            self.data['progress']['videos_completed'] /
            self.data['progress']['total_videos_target'] * 100
        )

        # Update timestamp
        self.data['timestamps']['last_updated'] = datetime.now().isoformat()

        # Save to disk
        self._save()

    def mark_video_failed(self, video_id, bucket, error):
        """Log failed video without stopping batch"""
        self.data['failed_videos'].append({
            'video_id': video_id,
            'bucket': bucket,
            'error': str(error),
            'timestamp': datetime.now().isoformat(),
            'retry_count': 0
        })

        # Update counts
        self.data['progress']['videos_failed'] += 1
        self.data['progress']['videos_remaining'] -= 1
        self.data['bucket_progress'][bucket]['videos_failed'] += 1

        self._save()

    def get_completed_video_ids(self):
        """Get set of all completed video IDs across buckets"""
        completed = set()
        for bucket_data in self.data['bucket_progress'].values():
            completed.update(bucket_data['completed_video_ids'])
        return completed

    def mark_completed(self):
        """Mark entire analysis as completed"""
        self.data['status'] = 'completed'
        self.data['timestamps']['completed_at'] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """Write checkpoint to disk"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)
```

---

### Batch Processing with Auto-Resume

```python
def run_batch_analysis(client_id, analysis_type, target, config, force=False):
    """Main batch processing with auto-resume"""

    checkpoint_path = f"/data/clients/{client_id}/{analysis_type}/{target}/checkpoints/analysis_state.json"

    # Check for existing checkpoint
    if os.path.exists(checkpoint_path) and not force:
        checkpoint = CheckpointManager(checkpoint_path)

        # Validate config matches
        if not checkpoint.validate_config(config):
            raise ValueError("Checkpoint config mismatch. Use --force to restart with new config.")

        # Auto-resume
        print(f"✓ Checkpoint detected: {checkpoint.data['progress']['videos_completed']}/{checkpoint.data['progress']['total_videos_target']} completed")
        print(f"→ Auto-resuming...")

        completed_ids = checkpoint.get_completed_video_ids()

    else:
        # Force restart or no checkpoint
        if force and os.path.exists(checkpoint_path):
            # Backup old checkpoint
            backup_path = checkpoint_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            shutil.copy(checkpoint_path, backup_path)
            print(f"→ Old checkpoint backed up to {backup_path}")

        # Create new checkpoint
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.initialize(client_id, analysis_type, target, config)
        completed_ids = set()

        print("→ Starting fresh analysis...")

    # Fetch videos via Apify
    all_videos = apify_scraper.fetch_videos(target, count=config['video_count'])

    # Bucket and select videos (Top 40 + Bottom 20 per bucket)
    bucketed_videos = bucket_and_select_videos(all_videos)

    # Filter out completed videos
    videos_to_process = [v for v in bucketed_videos if v['id'] not in completed_ids]

    print(f"→ Processing {len(videos_to_process)} videos...")

    # Process videos with checkpoint updates
    for video in videos_to_process:
        try:
            # Run RumiAI analysis
            result = rumiai_runner.process_video(video['id'])

            # Mark as completed in checkpoint
            checkpoint.mark_video_completed(video['id'], video['bucket'])

            print(f"✓ {video['id']} completed")

        except Exception as e:
            # Log failure but continue batch (skip-on-fail)
            checkpoint.mark_video_failed(video['id'], video['bucket'], str(e))
            print(f"✗ {video['id']} failed: {e}")
            continue

    # Mark analysis complete
    checkpoint.mark_completed()

    # Print summary
    print(f"\n✓ Analysis completed:")
    print(f"  - Total videos: {checkpoint.data['progress']['total_videos_target']}")
    print(f"  - Completed: {checkpoint.data['progress']['videos_completed']}")
    print(f"  - Failed: {checkpoint.data['progress']['videos_failed']}")
```

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

**Implementation**:
```python
def _load_or_create(self):
    if os.path.exists(self.path):
        try:
            with open(self.path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("✗ Checkpoint file corrupted")
            # Look for backup
            backup_files = glob.glob(self.path.replace('.json', '_backup_*.json'))
            if backup_files:
                latest_backup = sorted(backup_files)[-1]
                print(f"→ Found backup: {latest_backup}")
                print("→ Use --force to create fresh checkpoint")
            raise
    return self._create_new_checkpoint()
```

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

**Handling**:
```python
try:
    checkpoint.mark_video_completed(video_id, bucket)
except IOError as e:
    print(f"✗ CRITICAL: Cannot save checkpoint - {e}")
    print("→ Free up disk space and resume")
    sys.exit(1)  # Stop batch, preserve last saved checkpoint
```

---

## Testing Strategy

### Unit Tests

```python
def test_checkpoint_creation():
    """Test new checkpoint initialization"""
    checkpoint = CheckpointManager.create_new(
        client_id="test_client",
        analysis_type="hashtag",
        target="#test"
    )
    assert checkpoint.data['status'] == 'in_progress'
    assert checkpoint.data['progress']['videos_completed'] == 0

def test_checkpoint_resume():
    """Test loading existing checkpoint"""
    # Create checkpoint with some progress
    checkpoint = CheckpointManager(test_path)
    checkpoint.mark_video_completed("video_1", "bucket_0-3s")

    # Load checkpoint
    checkpoint2 = CheckpointManager(test_path)
    assert "video_1" in checkpoint2.get_completed_video_ids()

def test_checkpoint_config_validation():
    """Test config mismatch detection"""
    checkpoint = CheckpointManager.create_new(config={'video_count': 300})

    # Try to resume with different config
    with pytest.raises(ValueError):
        checkpoint.validate_config({'video_count': 250})
```

### Integration Tests

```python
def test_batch_with_interruption():
    """Test full batch with simulated interruption"""
    videos = generate_test_videos(100)

    # Start batch, interrupt at 50%
    batch = run_batch_analysis(videos[:50])

    # Resume batch
    batch = run_batch_analysis(videos, resume=True)

    # Verify no duplicate processing
    assert batch.videos_processed == 100
    assert batch.duplicate_count == 0
```

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