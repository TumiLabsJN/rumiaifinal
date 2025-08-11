#!/usr/bin/env python3
"""
Comprehensive analysis of ALL files that will be deleted in Phase 2
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def get_file_info(filepath: str) -> Tuple[int, str]:
    """Get line count and first line description"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            line_count = len(lines)
            
            # Try to get description from docstring
            content = ''.join(lines[:20])
            doc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if doc_match:
                desc = doc_match.group(1).strip().split('\n')[0][:80]
            else:
                # Get first comment
                for line in lines[:10]:
                    if line.strip().startswith('#') and not line.startswith('#!/'):
                        desc = line.strip()[1:].strip()[:80]
                        break
                else:
                    desc = "No description"
            
            return line_count, desc
    except:
        return 0, "Cannot read file"

def analyze_deletable_files():
    """Analyze all files that will be deleted"""
    
    categories = {
        'test_scripts_debug': [],
        'test_scripts_keep': [],
        'cleanup_scripts': [],
        'claude_files': [],
        'setup_files': [],
        'test_audio': [],
        'directories': [],
        'prompts': []
    }
    
    # 1. Test scripts to DELETE (debug/one-time)
    test_to_delete = [
        'test_analysis_structure.py',
        'test_extraction.py',
        'test_extraction_fixed.py',
        'test_failfast.py',
        'test_format_defense.py',
        'test_helper_solution.py',
        'test_helper_vs_timeline.py',
        'test_helpers.py',
        'test_which_wrapper.py',
        'test_root_cause.py',
        'test_unified_pipeline_e2e.py'
    ]
    
    for test_file in test_to_delete:
        if os.path.exists(test_file):
            lines, desc = get_file_info(test_file)
            categories['test_scripts_debug'].append((test_file, lines, desc))
    
    # 2. Test scripts to KEEP
    test_to_keep = [
        'test_python_only_e2e.py',
        'test_unified_ml_pipeline.py',
        'test_audio_energy.py',
        'test_ml_fixes.py',
        'test_p0_fixes.py',
        'test_whisper_cpp.py'
    ]
    
    for test_file in test_to_keep:
        if os.path.exists(test_file):
            lines, desc = get_file_info(test_file)
            categories['test_scripts_keep'].append((test_file, lines, desc))
    
    # 3. Cleanup/analysis scripts
    cleanup_patterns = [
        ('discovery_*.py', 'Discovery/analysis scripts'),
        ('remove_*.py', 'Removal scripts'),
        ('hardcode_*.py', 'Hardcoding scripts'),
        ('final_*.py', 'Final validation scripts'),
        ('scipy_compat.py', 'Scipy compatibility'),
        ('analyze_*.py', 'Analysis scripts'),
        ('check_*.py', 'Check scripts'),
        ('count_*.py', 'Counting scripts'),
        ('debug_*.py', 'Debug scripts'),
        ('trace_*.py', 'Trace scripts'),
        ('verify_*.py', 'Verification scripts'),
        ('investigate_*.py', 'Investigation scripts'),
        ('detect_*.py', 'Detection scripts'),
        ('simplify_*.py', 'Simplification scripts')
    ]
    
    for pattern, category in cleanup_patterns:
        for filepath in Path('.').glob(pattern):
            if 'smart_cleanup' not in str(filepath):  # Don't delete our cleanup script
                lines, desc = get_file_info(str(filepath))
                categories['cleanup_scripts'].append((str(filepath), lines, desc))
    
    # 4. Claude-specific files
    claude_files = [
        'rumiai_v2/core/models/prompt.py',
        'rumiai_v2/api/claude_client.py',
        'rumiai_v2/processors/prompt_builder.py',
        'rumiai_v2/processors/output_adapter.py',
        'rumiai_v2/processors/ml_data_extractor.py'
    ]
    
    for claude_file in claude_files:
        if os.path.exists(claude_file):
            lines, desc = get_file_info(claude_file)
            categories['claude_files'].append((claude_file, lines, desc))
    
    # 5. Setup/deployment files
    setup_files = [
        'setup.sh',
        'Dockerfile',
        '.env.example',
        'install_ml_dependencies.sh',
        'set_python_only_env.sh'
    ]
    
    for setup_file in setup_files:
        if os.path.exists(setup_file):
            if setup_file.endswith('.py'):
                lines, desc = get_file_info(setup_file)
            else:
                # For non-Python files, just count lines
                try:
                    with open(setup_file, 'r') as f:
                        lines = len(f.readlines())
                    desc = f"{setup_file.split('.')[-1].upper()} file"
                except:
                    lines = 0
                    desc = "Binary or unreadable"
            categories['setup_files'].append((setup_file, lines, desc))
    
    # 6. Test audio files
    test_audio = [
        'test_120s.wav',
        'test_30s.wav', 
        'test_3s.wav',
        'test_audio.wav'
    ]
    
    for audio_file in test_audio:
        if os.path.exists(audio_file):
            # Get file size instead of lines for audio
            size = os.path.getsize(audio_file)
            size_mb = size / (1024 * 1024)
            categories['test_audio'].append((audio_file, 0, f"Audio file ({size_mb:.1f} MB)"))
    
    # 7. Directories to delete
    dirs_to_check = [
        'tests/',
        'rumiai_v2/prompts/',
        'prompt_templates/'
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            # Count Python files and lines in directory
            py_files = list(Path(dir_path).rglob('*.py'))
            total_lines = 0
            for py_file in py_files:
                try:
                    with open(py_file, 'r') as f:
                        total_lines += len(f.readlines())
                except:
                    pass
            categories['directories'].append((dir_path, total_lines, f"Directory with {len(py_files)} Python files"))
    
    # 8. Check for prompt template files
    for prompt_file in Path('.').rglob('*prompt*.py'):
        if str(prompt_file) not in [f[0] for f in categories['claude_files']]:
            if 'rumiai_v2' in str(prompt_file):
                lines, desc = get_file_info(str(prompt_file))
                categories['prompts'].append((str(prompt_file), lines, desc))
    
    return categories

def print_analysis():
    """Print detailed analysis"""
    categories = analyze_deletable_files()
    
    print("=" * 80)
    print("COMPREHENSIVE PHASE 2 DELETION ANALYSIS")
    print("=" * 80)
    
    total_delete_lines = 0
    total_delete_files = 0
    total_keep_lines = 0
    total_keep_files = 0
    
    # Files to DELETE
    print("\n❌ FILES TO DELETE:")
    print("-" * 80)
    
    print("\n1. DEBUG/ONE-TIME TEST SCRIPTS:")
    for filepath, lines, desc in sorted(categories['test_scripts_debug']):
        print(f"   • {filepath:<35} ({lines:4} lines) - {desc}")
        total_delete_lines += lines
        total_delete_files += 1
    
    print("\n2. CLEANUP/ANALYSIS SCRIPTS:")
    for filepath, lines, desc in sorted(categories['cleanup_scripts']):
        print(f"   • {filepath:<35} ({lines:4} lines) - {desc}")
        total_delete_lines += lines
        total_delete_files += 1
    
    print("\n3. CLAUDE-SPECIFIC FILES:")
    for filepath, lines, desc in sorted(categories['claude_files']):
        print(f"   • {filepath:<35} ({lines:4} lines) - {desc}")
        total_delete_lines += lines
        total_delete_files += 1
    
    print("\n4. SETUP/DEPLOYMENT FILES:")
    for filepath, lines, desc in sorted(categories['setup_files']):
        print(f"   • {filepath:<35} ({lines:4} lines) - {desc}")
        total_delete_lines += lines
        total_delete_files += 1
    
    print("\n5. TEST AUDIO FILES:")
    for filepath, lines, desc in sorted(categories['test_audio']):
        print(f"   • {filepath:<35} - {desc}")
        total_delete_files += 1
    
    print("\n6. DIRECTORIES TO DELETE:")
    for dirpath, lines, desc in sorted(categories['directories']):
        print(f"   • {dirpath:<35} ({lines:4} lines) - {desc}")
        total_delete_lines += lines
        # Count files in directory
        if os.path.exists(dirpath):
            py_files = list(Path(dirpath).rglob('*.py'))
            total_delete_files += len(py_files)
    
    if categories['prompts']:
        print("\n7. ADDITIONAL PROMPT FILES:")
        for filepath, lines, desc in sorted(categories['prompts']):
            print(f"   • {filepath:<35} ({lines:4} lines) - {desc}")
            total_delete_lines += lines
            total_delete_files += 1
    
    # Files to KEEP
    print("\n" + "=" * 80)
    print("✅ FILES TO KEEP:")
    print("-" * 80)
    
    print("\nCRITICAL PYTHON-ONLY TEST SCRIPTS:")
    for filepath, lines, desc in sorted(categories['test_scripts_keep']):
        print(f"   • {filepath:<35} ({lines:4} lines) - {desc}")
        total_keep_lines += lines
        total_keep_files += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("-" * 80)
    print(f"Files to DELETE: {total_delete_files}")
    print(f"Lines to DELETE: {total_delete_lines:,}")
    print(f"Files to KEEP: {total_keep_files}")
    print(f"Lines to KEEP: {total_keep_lines:,}")
    print(f"\nTotal code reduction: ~{total_delete_lines:,} lines")
    print(f"Percentage reduction: ~{(total_delete_lines/(total_delete_lines+total_keep_lines)*100):.1f}%")
    print("=" * 80)
    
    # Generate markdown for finalcd2.md
    print("\n\nMARKDOWN FOR finalcd2.md:")
    print("=" * 80)
    print("```markdown")
    print("## EXACT FILES TO BE DELETED\n")
    
    print("### Test Scripts (Debug/One-time) - DELETE")
    for filepath, lines, desc in sorted(categories['test_scripts_debug']):
        print(f"- `{filepath}` ({lines} lines)")
    
    print("\n### Cleanup/Analysis Scripts - DELETE")
    for filepath, lines, desc in sorted(categories['cleanup_scripts']):
        print(f"- `{filepath}` ({lines} lines)")
    
    print("\n### Claude-specific Files - DELETE")
    for filepath, lines, desc in sorted(categories['claude_files']):
        print(f"- `{filepath}` ({lines} lines)")
    
    print("\n### Setup/Deployment Files - DELETE")
    for filepath, lines, desc in sorted(categories['setup_files']):
        print(f"- `{filepath}` ({lines} lines)")
    
    print("\n### Test Audio Files - DELETE")
    for filepath, lines, desc in sorted(categories['test_audio']):
        print(f"- `{filepath}`")
    
    print("\n### Directories - DELETE ENTIRE DIRECTORY")
    for dirpath, lines, desc in sorted(categories['directories']):
        print(f"- `{dirpath}` ({lines} lines total)")
    
    print("\n### Critical Test Scripts - KEEP")
    for filepath, lines, desc in sorted(categories['test_scripts_keep']):
        print(f"- `{filepath}` ({lines} lines) ✅")
    
    print("```")

if __name__ == "__main__":
    print_analysis()