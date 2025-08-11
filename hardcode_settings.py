#!/usr/bin/env python3
"""Hardcode all settings for zero-configuration operation"""

from pathlib import Path
import re
import sys

def hardcode_settings():
    """Hardcode all Python-only settings using robust regex replacements"""
    
    settings_file = Path('rumiai_v2/config/settings.py')
    if not settings_file.exists():
        print("❌ settings.py not found")
        return False
        
    try:
        content = settings_file.read_text()
        original_content = content  # Backup for validation
        
        # Define replacements with regex patterns that work regardless of line position
        replacements = [
            # Boolean settings - replace entire line while preserving indentation
            (r'(\s*)self\.use_python_only_processing\s*=.*$', 
             r'\1self.use_python_only_processing = True  # HARDCODED'),
            (r'(\s*)self\.use_ml_precompute\s*=.*$', 
             r'\1self.use_ml_precompute = True  # HARDCODED'),
            (r'(\s*)self\.temporal_markers_enabled\s*=.*$', 
             r'\1self.temporal_markers_enabled = True  # HARDCODED'),
            (r'(\s*)self\.enable_cost_monitoring\s*=.*$', 
             r'\1self.enable_cost_monitoring = True  # HARDCODED'),
            
            # String setting
            (r'(\s*)self\.output_format_version\s*=.*$', 
             r'\1self.output_format_version = "v2"  # HARDCODED'),
            
            # Dictionary entries - preserve indentation
            (r"(\s*)'creative_density':\s*os\.getenv.*$", 
             r"\1'creative_density': True,  # HARDCODED"),
            (r"(\s*)'emotional_journey':\s*os\.getenv.*$", 
             r"\1'emotional_journey': True,  # HARDCODED"),
            (r"(\s*)'person_framing':\s*os\.getenv.*$", 
             r"\1'person_framing': True,  # HARDCODED"),
            (r"(\s*)'scene_pacing':\s*os\.getenv.*$", 
             r"\1'scene_pacing': True,  # HARDCODED"),
            (r"(\s*)'speech_analysis':\s*os\.getenv.*$", 
             r"\1'speech_analysis': True,  # HARDCODED"),
            (r"(\s*)'visual_overlay_analysis':\s*os\.getenv.*$", 
             r"\1'visual_overlay_analysis': True,  # HARDCODED"),
            (r"(\s*)'metadata_analysis':\s*os\.getenv.*$", 
             r"\1'metadata_analysis': True  # HARDCODED"),
        ]
        
        # Apply replacements
        changes_made = 0
        for pattern, replacement in replacements:
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            if new_content != content:
                changes_made += 1
                content = new_content
        
        # Validate changes were made
        if changes_made == 0:
            print("⚠️ No settings were hardcoded - they may already be hardcoded")
            return True  # Not an error if already hardcoded
        
        # Verify critical settings are now hardcoded
        validations = [
            ('self.use_python_only_processing = True', 'Python-only mode'),
            ('self.use_ml_precompute = True', 'ML precompute'),
            ('self.output_format_version = "v2"', 'Output format v2'),
        ]
        
        for check, name in validations:
            if check not in content:
                print(f"❌ Failed to hardcode {name}")
                return False
        
        # Write back only if validations pass
        settings_file.write_text(content)
        print(f"✅ Successfully hardcoded {changes_made} settings")
        print("✅ All environment variables removed - settings are now hardcoded!")
        return True
        
    except Exception as e:
        print(f"❌ Error hardcoding settings: {e}")
        return False

if __name__ == "__main__":
    success = hardcode_settings()
    sys.exit(0 if success else 1)