"""
Solution 1: RAM Disk Implementation for FEAT
This shows the exact changes needed in emotion_detection_service.py
"""

import tempfile
import os
from pathlib import Path

# ==============================================================================
# ADD THIS SECTION AT THE TOP OF EmotionDetectionService.__init__
# ==============================================================================

class EmotionDetectionService:
    def __init__(self, gpu: bool = True):
        # ... existing initialization code ...

        # NEW: Setup RAM disk for temp files
        self.temp_dir = self._setup_ramdisk()
        if self.temp_dir:
            logger.info(f"✅ Using RAM disk for FEAT temp files: {self.temp_dir}")
        else:
            logger.warning("⚠️ RAM disk not available, using regular temp directory")

    def _setup_ramdisk(self) -> Path:
        """
        Setup RAM disk for temporary files to eliminate I/O bottleneck
        Returns Path to RAM disk directory or None if not available
        """
        # Option 1: Use /dev/shm (standard Linux shared memory)
        if os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK):
            ramdisk_path = Path('/dev/shm/feat_temp')
            try:
                ramdisk_path.mkdir(exist_ok=True)
                # Test write permissions
                test_file = ramdisk_path / 'test.txt'
                test_file.write_text('test')
                test_file.unlink()
                return ramdisk_path
            except Exception as e:
                logger.warning(f"Cannot use /dev/shm: {e}")

        # Option 2: Use /tmp if it's a tmpfs mount (check with df -h /tmp)
        tmp_path = Path('/tmp')
        try:
            # Check if /tmp is mounted as tmpfs (in memory)
            import subprocess
            result = subprocess.run(['df', '-T', '/tmp'],
                                  capture_output=True, text=True)
            if 'tmpfs' in result.stdout:
                feat_tmp = tmp_path / 'feat_temp'
                feat_tmp.mkdir(exist_ok=True)
                return feat_tmp
        except:
            pass

        # Option 3: Create explicit RAM disk (requires sudo initially)
        # This would need to be set up outside Python:
        # sudo mkdir -p /mnt/ramdisk
        # sudo mount -t tmpfs -o size=512M tmpfs /mnt/ramdisk
        if os.path.exists('/mnt/ramdisk') and os.access('/mnt/ramdisk', os.W_OK):
            ramdisk_path = Path('/mnt/ramdisk/feat_temp')
            ramdisk_path.mkdir(exist_ok=True)
            return ramdisk_path

        return None

# ==============================================================================
# MODIFY THE _detect_batch METHOD (around line 321)
# ==============================================================================

    def _detect_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Detect emotions in batch using FEAT (synchronous)
        NOW WITH RAM DISK OPTIMIZATION
        """
        import tempfile
        import os

        # FEAT expects image file paths, not numpy arrays
        temp_files = []
        try:
            # Save frames to temporary files
            for i, frame in enumerate(frames):
                # Convert BGR to RGB for saving
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # NEW: Use RAM disk directory if available
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=f'_frame_{i}.jpg',
                    delete=False,
                    dir=self.temp_dir  # ← THIS IS THE KEY CHANGE!
                )

                cv2.imwrite(temp_file.name, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                temp_files.append(temp_file.name)
                temp_file.close()

            # Run FEAT detection on file paths
            predictions = self.detector.detect_image(temp_files)

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass  # File already deleted or doesn't exist

        # ... rest of the method remains the same ...

# ==============================================================================
# OPTIONAL: Add cleanup method to clear RAM disk on exit
# ==============================================================================

    def cleanup(self):
        """Call this when shutting down to clean RAM disk"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            try:
                # Only remove files, not the directory itself
                for file in self.temp_dir.glob('*.jpg'):
                    file.unlink()
                logger.info(f"Cleaned up RAM disk: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup RAM disk: {e}")