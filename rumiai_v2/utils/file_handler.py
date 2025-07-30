"""
File handling utilities for RumiAI v2.

CRITICAL: All file operations must be atomic to prevent corruption.
"""
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..core.exceptions import FileSystemError

logger = logging.getLogger(__name__)


class FileHandler:
    """
    Handle file operations with atomic writes and validation.
    
    CRITICAL: Multiple processes may access same files.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, file_path: Path, data: Dict[str, Any], 
                  indent: int = 2, atomic: bool = True) -> None:
        """
        Save data to JSON file.
        
        CRITICAL: Uses atomic write to prevent corruption.
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if atomic:
                # Write to temporary file first
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    dir=file_path.parent,
                    delete=False,
                    suffix='.tmp'
                ) as tmp:
                    json.dump(data, tmp, indent=indent)
                    tmp_path = tmp.name
                
                # Atomic move
                shutil.move(tmp_path, file_path)
            else:
                # Direct write (faster but not atomic)
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=indent)
            
            logger.info(f"Saved JSON to {file_path}")
            
        except Exception as e:
            raise FileSystemError('save', str(file_path), str(e))
    
    def load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileSystemError('load', str(file_path), 'File not found')
        except json.JSONDecodeError as e:
            raise FileSystemError('load', str(file_path), f'Invalid JSON: {e}')
        except Exception as e:
            raise FileSystemError('load', str(file_path), str(e))
    
    def exists(self, file_path: Path) -> bool:
        """Check if file exists."""
        return file_path.exists()
    
    def get_path(self, *parts: str) -> Path:
        """Build path relative to base directory."""
        path = self.base_dir
        for part in parts:
            path = path / part
        return path
    
    def ensure_dir(self, dir_path: Path) -> None:
        """Ensure directory exists."""
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise FileSystemError('mkdir', str(dir_path), str(e))
    
    def list_files(self, pattern: str = "*", recursive: bool = False) -> list[Path]:
        """List files matching pattern."""
        if recursive:
            return list(self.base_dir.rglob(pattern))
        else:
            return list(self.base_dir.glob(pattern))
    
    def delete_file(self, file_path: Path, safe: bool = True) -> None:
        """
        Delete a file.
        
        Args:
            file_path: Path to file
            safe: If True, only delete if within base_dir
        """
        if safe and not str(file_path).startswith(str(self.base_dir)):
            raise FileSystemError(
                'delete',
                str(file_path),
                'Path outside base directory'
            )
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted {file_path}")
        except Exception as e:
            raise FileSystemError('delete', str(file_path), str(e))
    
    def move_file(self, src: Path, dst: Path, atomic: bool = True) -> None:
        """Move file from src to dst."""
        try:
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if atomic and src.parent == dst.parent:
                # Atomic rename in same directory
                src.rename(dst)
            else:
                # Copy then delete
                shutil.move(str(src), str(dst))
            
            logger.info(f"Moved {src} to {dst}")
            
        except Exception as e:
            raise FileSystemError('move', f"{src} -> {dst}", str(e))
    
    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            return file_path.stat().st_size
        except Exception as e:
            raise FileSystemError('stat', str(file_path), str(e))
    
    def cleanup_old_files(self, pattern: str, days: int = 7) -> int:
        """
        Clean up files older than specified days.
        
        Returns number of files deleted.
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        deleted = 0
        for file_path in self.list_files(pattern, recursive=True):
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    self.delete_file(file_path)
                    deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete old file {file_path}: {e}")
        
        logger.info(f"Cleaned up {deleted} old files matching '{pattern}'")
        return deleted