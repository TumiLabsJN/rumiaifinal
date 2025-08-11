#!/usr/bin/env python3
"""
Remove ALL Claude and dead code from production files
This must be run BEFORE the file deletion phase
"""

import re
from pathlib import Path

def clean_rumiai_runner():
    """Remove all Claude methods and references from rumiai_runner.py"""
    
    file_path = Path('scripts/rumiai_runner.py')
    content = file_path.read_text()
    original_lines = len(content.split('\n'))
    
    # 1. Remove Claude method definitions entirely
    # Find and remove entire method bodies
    claude_methods = [
        '_run_claude_prompts',
        '_run_claude_prompts_v2', 
        '_save_prompt_result',
        '_format_claude_response',
        '_handle_claude_error'
    ]
    
    for method in claude_methods:
        # Match entire method from 'def method' to next 'def ' or class end
        pattern = rf'(\n    async def {method}.*?)(?=\n    def |\n    async def |\nclass |\n\n[^\s]|\Z)'
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        pattern = rf'(\n    def {method}.*?)(?=\n    def |\n    async def |\nclass |\n\n[^\s]|\Z)'
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # 2. Replace Claude prompt calls with precompute
    # Find lines that call Claude prompts and replace with precompute
    content = re.sub(
        r'.*print.*running_claude_prompts.*\n',
        '            print("📊 running_precompute_functions... (70%)")\n',
        content
    )
    
    # Replace the entire Claude conditional block
    pattern = r'# Use v2 if feature flag is enabled.*?prompt_results = await self\._run_claude_prompts\(unified_analysis\)'
    replacement = '''# Run precompute functions instead of Claude
            prompt_results = {}
            for func_name, func in COMPUTE_FUNCTIONS.items():
                try:
                    result = func(unified_analysis.to_dict())
                    prompt_results[func_name] = result
                except Exception as e:
                    logger.error(f"Precompute {func_name} failed: {e}")
                    prompt_results[func_name] = {}'''
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 3. Remove any remaining Claude references
    # Remove lines with Claude comments
    content = re.sub(r'.*# Step \d+: Run Claude prompts.*\n', '', content)
    content = re.sub(r'.*#.*Claude.*\n', '', content, flags=re.IGNORECASE)
    
    # 4. Clean up imports - remove any Claude/prompt related imports
    content = re.sub(r'from rumiai_v2\.core\.models import.*PromptType.*\n', 
                     'from rumiai_v2.core.models import VideoMetadata\n', content)
    content = re.sub(r'.*PromptBatch.*', '', content)
    
    # 5. Fix the imports to ensure COMPUTE_FUNCTIONS is imported
    if 'from rumiai_v2.processors import' in content and 'COMPUTE_FUNCTIONS' not in content:
        content = re.sub(
            r'(from rumiai_v2\.processors import.*?)(\n)',
            r'\1, get_compute_function, COMPUTE_FUNCTIONS\2',
            content
        )
    
    # Write back
    file_path.write_text(content)
    new_lines = len(content.split('\n'))
    print(f"✅ Cleaned rumiai_runner.py: removed {original_lines - new_lines} lines")
    
    return original_lines - new_lines

def clean_api_init():
    """Remove ClaudeClient from API __init__.py"""
    
    file_path = Path('rumiai_v2/api/__init__.py')
    if not file_path.exists():
        return 0
        
    content = file_path.read_text()
    original = content
    
    # Remove ClaudeClient from imports
    content = re.sub(r'from \.claude_client import ClaudeClient\n', '', content)
    
    # Remove from __all__
    content = re.sub(r"'ClaudeClient',?\s*\n", '', content)
    
    # Clean up any double commas or trailing commas
    content = re.sub(r',\s*,', ',', content)
    content = re.sub(r",\s*\]", ']', content)
    
    if content != original:
        file_path.write_text(content)
        print("✅ Cleaned rumiai_v2/api/__init__.py")
        return 10
    return 0

def clean_settings():
    """Remove Claude settings and prompt templates"""
    
    file_path = Path('rumiai_v2/config/settings.py')
    if not file_path.exists():
        return 0
    
    content = file_path.read_text()
    original_lines = len(content.split('\n'))
    
    # Remove prompt template loading
    content = re.sub(
        r'# Load prompt templates.*?def get_prompt_template.*?\n.*?\n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Remove Claude-related settings
    content = re.sub(r'.*self\.claude.*\n', '', content)
    content = re.sub(r'.*self\.prompt.*\n', '', content)
    content = re.sub(r'.*CLAUDE.*\n', '', content)
    
    # Remove Claude validation
    content = re.sub(
        r'if not self\.claude_api_key:.*?\n.*?\n',
        '',
        content
    )
    
    file_path.write_text(content)
    new_lines = len(content.split('\n'))
    print(f"✅ Cleaned settings.py: removed {original_lines - new_lines} lines")
    return original_lines - new_lines

def remove_dead_conditionals():
    """Remove dead conditional code that's always True/False"""
    
    total_removed = 0
    
    # Files that might have dead conditionals
    files_to_check = [
        'scripts/rumiai_runner.py',
        'rumiai_v2/processors/temporal_markers.py',
        'rumiai_v2/core/models/timestamp.py',
        'rumiai_v2/core/models/timeline.py',
        'rumiai_v2/core/models/analysis.py'
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            continue
            
        content = path.read_text()
        original = content
        
        # Remove use_ml_precompute conditionals (always True)
        pattern = r'if self\.settings\.use_ml_precompute:.*?else:.*?(?=\n\S|\Z)'
        content = re.sub(pattern, 
                        lambda m: m.group().split('else:')[0].replace('if self.settings.use_ml_precompute:', ''),
                        content, flags=re.DOTALL)
        
        # Remove legacy_mode parameters (always True)
        content = re.sub(r',?\s*legacy_mode:\s*bool\s*=\s*True', '', content)
        content = re.sub(r'if legacy_mode:.*?else:.*?(?=\n\S|\Z)',
                        lambda m: m.group().split('else:')[0].replace('if legacy_mode:', ''),
                        content, flags=re.DOTALL)
        
        if content != original:
            path.write_text(content)
            removed = len(original.split('\n')) - len(content.split('\n'))
            total_removed += removed
            print(f"✅ Cleaned {file_path}: removed {removed} lines of dead conditionals")
    
    return total_removed

def main():
    print("=" * 70)
    print("REMOVING ALL CLAUDE AND DEAD CODE FROM PRODUCTION")
    print("=" * 70)
    
    total_lines_removed = 0
    
    # 1. Clean rumiai_runner.py
    print("\n[1/4] Cleaning rumiai_runner.py...")
    total_lines_removed += clean_rumiai_runner()
    
    # 2. Clean API init
    print("\n[2/4] Cleaning API __init__.py...")
    total_lines_removed += clean_api_init()
    
    # 3. Clean settings
    print("\n[3/4] Cleaning settings.py...")
    total_lines_removed += clean_settings()
    
    # 4. Remove dead conditionals
    print("\n[4/4] Removing dead conditionals...")
    total_lines_removed += remove_dead_conditionals()
    
    print("\n" + "=" * 70)
    print(f"✅ CLEANUP COMPLETE!")
    print(f"📊 Total lines removed: {total_lines_removed}")
    print("=" * 70)
    print("\nNow run: python3 verify_production_safety.py")
    print("If it passes, then run: python3 smart_cleanup_phase2.py --execute")

if __name__ == "__main__":
    main()