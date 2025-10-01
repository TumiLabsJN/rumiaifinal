# Checkpoint Resume System - Technical Implementation

**Related HLD**: [MLCheckpointResume.md](./MLCheckpointResume.md)
**Status**: Implementation Ready
**Last Updated**: 2025-10-01

---

## Overview

This document contains the technical implementation for the Checkpoint Resume System. For high-level design, business context, and decision rationale, see [MLCheckpointResume.md](./MLCheckpointResume.md).

**Source**: Code extracted from MLAnalysisModeTI.md Section 4 - Checkpoint Integration

---

## CheckpointManager Implementation

### File Location

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/checkpoint_manager.py` (NEW)

### Complete Implementation

```python
"""
Checkpoint management for ML batch processing
Enables resume after failures without re-processing completed videos
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint state for batch video processing.

    Stores:
    - Configuration (must match on resume)
    - Completed video IDs
    - Current position
    - Processing metadata
    """

    def __init__(self, client_id: str, analysis_type: str, target: str):
        """
        Initialize checkpoint manager.

        Args:
            client_id: Client identifier
            analysis_type: "hashtag", "competitor", or "creator"
            target: #hashtag or @handle
        """
        self.client_id = client_id
        self.analysis_type = analysis_type
        self.target = target

        # Checkpoint directory
        self.checkpoint_dir = Path("data") / "clients" / client_id / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint files
        safe_target = target.replace("#", "").replace("@", "")
        self.checkpoint_file = self.checkpoint_dir / f"{analysis_type}_{safe_target}.json"
        self.completed_file = self.checkpoint_dir / f"{analysis_type}_{safe_target}_completed.jsonl"

    def save_config(self, config: Dict[str, Any]):
        """
        Save initial configuration.

        Must be called before processing starts.
        Validates configuration on resume.
        """
        checkpoint_data = {
            "client_id": self.client_id,
            "analysis_type": self.analysis_type,
            "target": self.target,
            "config": config,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_processed": 0
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint initialized: {self.checkpoint_file}")

    def validate_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Validate that new configuration matches existing checkpoint.

        Returns:
            True if configs match, False otherwise

        Raises:
            ValueError if critical mismatch detected
        """
        if not self.checkpoint_file.exists():
            return True  # No checkpoint, validation passes

        with open(self.checkpoint_file) as f:
            checkpoint = json.load(f)

        old_config = checkpoint['config']

        # Critical fields that must match
        critical_fields = ['video_count', 'date_filter', 'analysis_mode']

        mismatches = []
        for field in critical_fields:
            if old_config.get(field) != new_config.get(field):
                mismatches.append(
                    f"  - {field}: checkpoint={old_config.get(field)}, new={new_config.get(field)}"
                )

        if mismatches:
            error_msg = (
                f"Cannot resume: checkpoint configuration mismatch!\n"
                f"{chr(10).join(mismatches)}\n"
                f"Use --force to discard checkpoint and restart."
            )
            raise ValueError(error_msg)

        logger.info("✓ Checkpoint configuration validated")
        return True

    def save_progress(self, video_id: str, bucket: str, position: int, features: Dict[str, Any]):
        """
        Save progress after successfully processing a video.

        Args:
            video_id: TikTok video ID
            bucket: Duration bucket (e.g., "bucket_13-18s")
            position: Sequential position in processing (0-indexed)
            features: Extracted features for this video
        """
        # Append to completed videos log (JSONL format)
        with open(self.completed_file, 'a') as f:
            f.write(json.dumps({
                "position": position,
                "video_id": video_id,
                "bucket": bucket,
                "features": features,
                "timestamp": datetime.now().isoformat()
            }) + '\n')

        # Update checkpoint with latest position
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)
        else:
            checkpoint = {
                "client_id": self.client_id,
                "analysis_type": self.analysis_type,
                "target": self.target
            }

        checkpoint.update({
            "last_position": position,
            "last_video_id": video_id,
            "last_bucket": bucket,
            "total_processed": position + 1,
            "last_updated": datetime.now().isoformat()
        })

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def get_resume_point(self) -> tuple[int, Optional[str]]:
        """
        Get position to resume from.

        Returns:
            (position, last_bucket) - Position to start from, last bucket processed
        """
        if not self.checkpoint_file.exists():
            return 0, None

        with open(self.checkpoint_file) as f:
            checkpoint = json.load(f)

        last_position = checkpoint.get('last_position', -1)
        last_bucket = checkpoint.get('last_bucket')

        # Resume from next position
        return last_position + 1, last_bucket

    def load_completed_features(self) -> list[Dict[str, Any]]:
        """
        Load all previously processed features for ML training.

        Returns:
            List of feature dictionaries
        """
        if not self.completed_file.exists():
            return []

        features = []
        with open(self.completed_file) as f:
            for line in f:
                video_data = json.loads(line)
                features.append(video_data["features"])

        return features

    def clear_checkpoint(self):
        """
        Clear checkpoint (force restart).

        Called when --force flag is used.
        """
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info(f"Removed checkpoint: {self.checkpoint_file}")

        if self.completed_file.exists():
            self.completed_file.unlink()
            logger.info(f"Removed completed log: {self.completed_file}")
```

---

## Integration Example

### Usage in ML Batch Processing

```python
"""
Example: Using CheckpointManager in batch processing workflow
"""
import asyncio
from pathlib import Path

from rumiai_v2.processors.checkpoint_manager import CheckpointManager
from rumiai_v2.api.apify_client import ApifyClient


async def process_hashtag_with_checkpoint(
    client_id: str,
    hashtag: str,
    video_count: int = 300,
    date_filter: str = "last_90_days",
    analysis_mode: str = "top",
    force_restart: bool = False
):
    """
    Hashtag analysis with checkpoint/resume support.
    """

    # 1. Initialize checkpoint manager
    checkpoint = CheckpointManager(client_id, "hashtag", hashtag)

    if force_restart:
        checkpoint.clear_checkpoint()

    # 2. Check for existing checkpoint
    resume_position, last_bucket = checkpoint.get_resume_point()

    if resume_position > 0:
        print(f"✓ Resuming from position {resume_position}")
        print(f"✓ Loading {resume_position} completed videos from checkpoint")

        # Validate config matches
        new_config = {
            "video_count": video_count,
            "date_filter": date_filter,
            "analysis_mode": analysis_mode
        }
        checkpoint.validate_config(new_config)
    else:
        print("Starting fresh batch processing")

        # Save initial config
        config = {
            "video_count": video_count,
            "date_filter": date_filter,
            "analysis_mode": analysis_mode
        }
        checkpoint.save_config(config)

    # 3. Process videos (example - simplified)
    videos_to_process = []  # TODO: Get from Apify, bucket, filter completed

    for position, video in enumerate(videos_to_process):
        if position < resume_position:
            continue  # Skip already completed

        try:
            # Run RumiAI analysis
            features = await analyze_video(video)

            # Save progress immediately after success
            checkpoint.save_progress(
                video_id=video.video_id,
                bucket=video.bucket,
                position=position,
                features=features
            )

            print(f"✓ Completed {position + 1}/{len(videos_to_process)}: {video.video_id}")

        except Exception as e:
            # Fail fast - stop on error for debugging
            print(f"✗ Failed at position {position}, video {video.video_id}")
            print(f"Error: {e}")
            print(f"To resume after fix: run with same parameters")
            raise

    print("\n✅ Batch processing complete!")

    # 4. Load all features for ML training
    all_features = checkpoint.load_completed_features()
    print(f"Loaded {len(all_features)} feature sets for ML training")

    return all_features
```

---

## Key Features

### 1. Analysis Mode Validation

**Critical Fields Validated**:
- `video_count`: Number of videos to scrape
- `date_filter`: Date constraint (e.g., "last_90_days")
- `analysis_mode`: "top" or "recent"

**Why This Matters**:
- Prevents resuming with different mode than started with
- Top mode vs recent mode fetch different videos
- Mixing modes would corrupt ML training data

**Example Error**:
```
Cannot resume: checkpoint configuration mismatch!
  - analysis_mode: checkpoint=top, new=recent
Use --force to discard checkpoint and restart.
```

### 2. JSONL Format for Completed Videos

**Benefits**:
- Append-only (safe for concurrent writes)
- One video per line (easy to parse)
- Can tail file for real-time progress monitoring
- Survives partial writes (last line may be corrupt, but rest intact)

**File Structure**:
```jsonl
{"position": 0, "video_id": "123", "bucket": "bucket_13-18s", "features": {...}, "timestamp": "..."}
{"position": 1, "video_id": "124", "bucket": "bucket_13-18s", "features": {...}, "timestamp": "..."}
{"position": 2, "video_id": "125", "bucket": "bucket_18-33s", "features": {...}, "timestamp": "..."}
```

### 3. Fail-Fast Architecture

**No Skip-on-Fail**:
- Errors raise exceptions (stop processing)
- Developer investigates and fixes bug
- Resume continues from exact failure point

**Why**:
- Bugs caught immediately (not after 6 hours)
- Clean data (no partial/corrupted features)
- Aligns with development workflow

---

## Testing

### Unit Tests

```python
"""
tests/test_checkpoint_manager.py
"""
import pytest
from pathlib import Path
import json

from rumiai_v2.processors.checkpoint_manager import CheckpointManager


def test_checkpoint_initialization(tmp_path):
    """Test checkpoint manager initialization."""
    checkpoint = CheckpointManager(
        client_id="test_client",
        analysis_type="hashtag",
        target="#test"
    )

    # Files should not exist yet
    assert not checkpoint.checkpoint_file.exists()
    assert not checkpoint.completed_file.exists()


def test_save_and_load_config(tmp_path):
    """Test config save and validation."""
    checkpoint = CheckpointManager("test_client", "hashtag", "#test")

    config = {
        "video_count": 300,
        "date_filter": "last_90_days",
        "analysis_mode": "top"
    }

    checkpoint.save_config(config)

    # File should exist
    assert checkpoint.checkpoint_file.exists()

    # Validate with same config (should pass)
    assert checkpoint.validate_config(config) is True


def test_config_mismatch_detection(tmp_path):
    """Test config validation detects mismatches."""
    checkpoint = CheckpointManager("test_client", "hashtag", "#test")

    original_config = {
        "video_count": 300,
        "date_filter": "last_90_days",
        "analysis_mode": "top"
    }
    checkpoint.save_config(original_config)

    # Try to resume with different analysis_mode
    new_config = {
        "video_count": 300,
        "date_filter": "last_90_days",
        "analysis_mode": "recent"  # Changed!
    }

    with pytest.raises(ValueError, match="checkpoint configuration mismatch"):
        checkpoint.validate_config(new_config)


def test_save_progress(tmp_path):
    """Test saving video progress."""
    checkpoint = CheckpointManager("test_client", "hashtag", "#test")

    features = {"word_count": 15, "energy_level": 0.7}

    checkpoint.save_progress(
        video_id="video_123",
        bucket="bucket_13-18s",
        position=0,
        features=features
    )

    # Checkpoint file should exist
    assert checkpoint.checkpoint_file.exists()
    assert checkpoint.completed_file.exists()

    # Verify checkpoint data
    with open(checkpoint.checkpoint_file) as f:
        data = json.load(f)

    assert data["last_position"] == 0
    assert data["last_video_id"] == "video_123"
    assert data["total_processed"] == 1


def test_get_resume_point(tmp_path):
    """Test resume point calculation."""
    checkpoint = CheckpointManager("test_client", "hashtag", "#test")

    # No checkpoint - should resume from start
    position, bucket = checkpoint.get_resume_point()
    assert position == 0
    assert bucket is None

    # Save progress
    checkpoint.save_progress("video_1", "bucket_13-18s", 0, {})
    checkpoint.save_progress("video_2", "bucket_13-18s", 1, {})

    # Should resume from next position
    position, bucket = checkpoint.get_resume_point()
    assert position == 2  # Last was 1, so resume from 2
    assert bucket == "bucket_13-18s"


def test_load_completed_features(tmp_path):
    """Test loading all completed features."""
    checkpoint = CheckpointManager("test_client", "hashtag", "#test")

    # Save multiple videos
    checkpoint.save_progress("v1", "bucket_13-18s", 0, {"word_count": 10})
    checkpoint.save_progress("v2", "bucket_13-18s", 1, {"word_count": 20})
    checkpoint.save_progress("v3", "bucket_18-33s", 2, {"word_count": 30})

    # Load all features
    features = checkpoint.load_completed_features()

    assert len(features) == 3
    assert features[0]["word_count"] == 10
    assert features[1]["word_count"] == 20
    assert features[2]["word_count"] == 30


def test_clear_checkpoint(tmp_path):
    """Test checkpoint clearing (force restart)."""
    checkpoint = CheckpointManager("test_client", "hashtag", "#test")

    # Create checkpoint
    checkpoint.save_config({"video_count": 300})
    checkpoint.save_progress("v1", "bucket_13-18s", 0, {})

    # Files should exist
    assert checkpoint.checkpoint_file.exists()
    assert checkpoint.completed_file.exists()

    # Clear checkpoint
    checkpoint.clear_checkpoint()

    # Files should be removed
    assert not checkpoint.checkpoint_file.exists()
    assert not checkpoint.completed_file.exists()
```

---

## Related Documentation

- **High-Level Design**: [MLCheckpointResume.md](./MLCheckpointResume.md)
- **Analysis Mode Integration**: [MLAnalysisMode.md](./MLAnalysisMode.md)
- **ML Planning**: [MLPlanning.md](../../MLPlanning.md)
