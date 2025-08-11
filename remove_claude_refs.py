#!/usr/bin/env python3
"""Remove all Claude references from the codebase"""

import re
from pathlib import Path
import sys

def remove_claude_from_runner():
    """Remove all Claude references from rumiai_runner.py"""
    
    runner_file = Path('scripts/rumiai_runner.py')
    if not runner_file.exists():
        print("❌ scripts/rumiai_runner.py not found")
        return False
        
    try:
        content = runner_file.read_text()
        original_content = content  # Backup
        
        # Remove ALL Claude-related imports
        content = re.sub(r'from rumiai_v2\.api import.*?ClaudeClient.*?\n', '', content)
        content = re.sub(r'from rumiai_v2\.processors import.*?OutputAdapter.*?\n', '', content)
        content = re.sub(r'from rumiai_v2\.processors import.*?PromptBuilder.*?\n', '', content)
        content = re.sub(r'from rumiai_v2\.processors import.*?MLDataExtractor.*?\n', '', content)
        content = re.sub(r'from rumiai_v2\.prompts import.*?\n', '', content)
        content = re.sub(r'import.*?claude.*?\n', '', content, flags=re.IGNORECASE)
        
        # Remove ALL Claude object instantiations
        content = re.sub(r'.*?self\.claude\s*=.*?\n', '', content)
        content = re.sub(r'.*?self\.output_adapter\s*=.*?\n', '', content)
        content = re.sub(r'.*?self\.prompt_builder\s*=.*?\n', '', content)
        content = re.sub(r'.*?self\.prompt_manager\s*=.*?\n', '', content)
        content = re.sub(r'.*?self\.ml_extractor\s*=.*?\n', '', content)
        
        # Remove any remaining references to Claude objects
        content = re.sub(r'self\.claude\.', '# Removed Claude call: self.claude.', content)
        content = re.sub(r'self\.output_adapter\.', '# Removed adapter call: self.output_adapter.', content)
        content = re.sub(r'self\.prompt_builder\.', '# Removed prompt builder: self.prompt_builder.', content)
        
        # Write back cleaned content
        runner_file.write_text(content)
        print("✅ Removed ALL Claude/Legacy references from rumiai_runner.py")
        
        # Also clean up any other Python files that might import Claude
        for py_file in Path('rumiai_v2').rglob('*.py'):
            if 'claude' not in str(py_file):  # Don't process files we're about to delete
                try:
                    file_content = py_file.read_text()
                    original = file_content
                    
                    # Remove Claude imports
                    file_content = re.sub(r'from.*?claude.*?import.*?\n', '', file_content, flags=re.IGNORECASE)
                    file_content = re.sub(r'import.*?claude.*?\n', '', file_content, flags=re.IGNORECASE)
                    
                    # Remove OutputAdapter imports
                    file_content = re.sub(r'from.*?output_adapter.*?import.*?\n', '', file_content, flags=re.IGNORECASE)
                    
                    if file_content != original:
                        py_file.write_text(file_content)
                        print(f"  ✓ Cleaned {py_file}")
                except Exception as e:
                    print(f"  ⚠️ Could not clean {py_file}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error removing Claude references: {e}")
        return False

if __name__ == "__main__":
    success = remove_claude_from_runner()
    sys.exit(0 if success else 1)