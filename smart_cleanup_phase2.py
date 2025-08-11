#!/usr/bin/env python3
"""
Phase 2 SMART Cleanup - Selective removal of dead code
Preserves critical Python-only tests while removing debug scripts
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple

class SmartCleanup:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.stats = {
            'files_deleted': 0,
            'lines_removed': 0,
            'files_preserved': 0,
            'lines_preserved': 0
        }
        
    def log(self, message: str, level="INFO"):
        """Log messages with level"""
        prefix = "🔍" if self.dry_run else "✅"
        print(f"{prefix} [{level}] {message}")
        
    def delete_file(self, filepath: str) -> int:
        """Delete a file and return line count"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    lines = len(f.readlines())
                
                if not self.dry_run:
                    os.remove(filepath)
                    self.log(f"Deleted {filepath} ({lines} lines)")
                else:
                    self.log(f"Would delete {filepath} ({lines} lines)")
                    
                self.stats['files_deleted'] += 1
                self.stats['lines_removed'] += lines
                return lines
        except Exception as e:
            self.log(f"Error deleting {filepath}: {e}", "ERROR")
        return 0
    
    def delete_directory(self, dirpath: str) -> int:
        """Delete a directory and return total line count"""
        total_lines = 0
        if os.path.exists(dirpath):
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r') as f:
                            total_lines += len(f.readlines())
            
            if not self.dry_run:
                shutil.rmtree(dirpath)
                self.log(f"Deleted directory {dirpath} ({total_lines} lines)")
            else:
                self.log(f"Would delete directory {dirpath} ({total_lines} lines)")
                
            self.stats['files_deleted'] += len(list(Path(dirpath).rglob('*.py')))
            self.stats['lines_removed'] += total_lines
        return total_lines
    
    def clean_test_files(self):
        """Selectively remove debug test files, keep critical ones"""
        print("\n📄 PHASE 1: Cleaning Test Files")
        print("=" * 50)
        
        # Test files to DELETE (debug/one-time scripts)
        debug_tests = [
            'test_analysis_structure.py',
            'test_extraction.py', 
            'test_extraction_fixed.py',
            'test_failfast.py',
            'test_format_defense.py',
            'test_helper_solution.py',
            'test_helper_vs_timeline.py',
            'test_helpers.py',
            'test_which_wrapper.py',
            'test_root_cause.py',  # If exists
            'test_unified_pipeline_e2e.py'  # Has Claude mocks
        ]
        
        # Test files to KEEP (critical for Python-only)
        keep_tests = [
            'test_python_only_e2e.py',
            'test_unified_ml_pipeline.py',
            'test_audio_energy.py',
            'test_ml_fixes.py',
            'test_p0_fixes.py',
            'test_whisper_cpp.py'
        ]
        
        # Delete debug tests
        for test_file in debug_tests:
            self.delete_file(test_file)
            
        # Report on preserved tests
        for test_file in keep_tests:
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    lines = len(f.readlines())
                self.log(f"Preserved {test_file} ({lines} lines)", "KEEP")
                self.stats['files_preserved'] += 1
                self.stats['lines_preserved'] += lines
    
    def clean_directories(self):
        """Remove entire directories that are no longer needed"""
        print("\n📁 PHASE 2: Removing Obsolete Directories")
        print("=" * 50)
        
        dirs_to_delete = [
            'tests/',  # Unit tests with Claude mocks
            'rumiai_v2/prompts/',  # Prompt templates
            'prompt_templates/',  # More prompts
        ]
        
        for directory in dirs_to_delete:
            self.delete_directory(directory)
    
    def clean_scripts(self):
        """Remove one-time cleanup and analysis scripts"""
        print("\n🔧 PHASE 3: Removing One-time Scripts")
        print("=" * 50)
        
        patterns = [
            'discovery_*.py',
            'remove_*.py',
            'hardcode_*.py',
            'final_*.py',
            'scipy_compat.py',
            'analyze_*.py',
            'check_*.py',
            'count_*.py',
            'debug_*.py',
            'trace_*.py',
            'verify_*.py',
            'investigate_*.py',
            'detect_*.py',
            'simplify_*.py'
        ]
        
        for pattern in patterns:
            for filepath in Path('.').glob(pattern):
                # Skip this cleanup script itself
                if 'smart_cleanup' not in str(filepath):
                    self.delete_file(str(filepath))
    
    def clean_claude_references(self):
        """Remove Claude-specific files"""
        print("\n🚫 PHASE 4: Removing Claude-specific Files")
        print("=" * 50)
        
        claude_files = [
            'rumiai_v2/core/models/prompt.py',
            'rumiai_v2/api/claude_client.py',
            'rumiai_v2/processors/prompt_builder.py',
            'rumiai_v2/processors/output_adapter.py'
        ]
        
        for filepath in claude_files:
            self.delete_file(filepath)
    
    def clean_test_audio(self):
        """Remove test audio files"""
        print("\n🔊 PHASE 5: Removing Test Audio Files")
        print("=" * 50)
        
        test_audio = [
            'test_120s.wav',
            'test_30s.wav',
            'test_3s.wav',
            'test_audio.wav'
        ]
        
        for audio_file in test_audio:
            if os.path.exists(audio_file):
                if not self.dry_run:
                    os.remove(audio_file)
                    self.log(f"Deleted {audio_file}")
                else:
                    self.log(f"Would delete {audio_file}")
                self.stats['files_deleted'] += 1
    
    def clean_setup_files(self):
        """Remove obsolete setup and deployment files"""
        print("\n⚙️ PHASE 6: Removing Setup Files")
        print("=" * 50)
        
        setup_files = [
            'setup.sh',
            'Dockerfile',
            '.env.example',
            'install_ml_dependencies.sh',
            'set_python_only_env.sh'
        ]
        
        for setup_file in setup_files:
            if os.path.exists(setup_file):
                if not self.dry_run:
                    os.remove(setup_file)
                    self.log(f"Deleted {setup_file}")
                else:
                    self.log(f"Would delete {setup_file}")
                self.stats['files_deleted'] += 1
    
    def print_summary(self):
        """Print cleanup summary"""
        print("\n" + "=" * 60)
        print("🎉 SMART CLEANUP SUMMARY")
        print("=" * 60)
        print(f"Files deleted: {self.stats['files_deleted']}")
        print(f"Lines removed: {self.stats['lines_removed']:,}")
        print(f"Files preserved: {self.stats['files_preserved']}")  
        print(f"Lines preserved: {self.stats['lines_preserved']:,}")
        print(f"\nTotal reduction: ~{self.stats['lines_removed']:,} lines")
        print(f"Critical tests preserved: ~{self.stats['lines_preserved']:,} lines")
        
        if self.dry_run:
            print("\n⚠️  This was a DRY RUN - no files were actually deleted")
            print("Run with --execute to perform actual cleanup")
    
    def run(self):
        """Execute the smart cleanup"""
        print("=" * 60)
        print("PHASE 2: SMART CLEANUP - Selective Dead Code Removal")
        print("=" * 60)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be deleted\n")
        else:
            print("⚠️  EXECUTING CLEANUP - Files will be deleted!\n")
            
        self.clean_test_files()
        self.clean_directories()
        self.clean_scripts()
        self.clean_claude_references()
        self.clean_test_audio()
        self.clean_setup_files()
        self.print_summary()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart cleanup for Phase 2')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually delete files (default is dry run)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be deleted without deleting')
    
    args = parser.parse_args()
    
    # If --execute is specified, it's not a dry run
    dry_run = not args.execute
    
    cleanup = SmartCleanup(dry_run=dry_run)
    cleanup.run()
    
    if dry_run:
        print("\n💡 TIP: Run with --execute to perform the actual cleanup:")
        print("   python3 smart_cleanup_phase2.py --execute")


if __name__ == "__main__":
    main()