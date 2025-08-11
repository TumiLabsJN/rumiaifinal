#!/usr/bin/env python3
"""
Production Safety Verification Script
Ensures no file marked for deletion is actually used in production flow
"""

import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List

class ProductionFlowTracer:
    def __init__(self):
        self.production_imports = set()
        self.production_files = set()
        self.files_to_delete = set()
        self.critical_errors = []
        self.claude_js_remnants = []
        
    def trace_production_flow(self):
        """Trace all imports and dependencies from rumiai_runner.py"""
        print("=" * 70)
        print("PRODUCTION FLOW DEPENDENCY ANALYSIS")
        print("=" * 70)
        
        # Start from the main entry point
        entry_point = 'scripts/rumiai_runner.py'
        self.production_files.add(entry_point)
        
        # Recursively trace all imports
        to_process = [entry_point]
        processed = set()
        
        while to_process:
            current_file = to_process.pop(0)
            if current_file in processed:
                continue
            processed.add(current_file)
            
            imports = self.extract_imports(current_file)
            for imp in imports:
                # Convert import to file path
                file_paths = self.import_to_filepath(imp)
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        self.production_files.add(file_path)
                        if file_path not in processed:
                            to_process.append(file_path)
        
        print(f"\n✅ Found {len(self.production_files)} production files")
        return self.production_files
    
    def extract_imports(self, filepath):
        """Extract all imports from a Python file"""
        imports = set()
        try:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except Exception as e:
            pass  # Skip files we can't parse
        
        return imports
    
    def import_to_filepath(self, import_name):
        """Convert import statement to possible file paths"""
        paths = []
        
        # Handle rumiai_v2 imports
        if import_name.startswith('rumiai_v2'):
            path = import_name.replace('.', '/') + '.py'
            paths.append(path)
            # Also check for __init__.py in package
            paths.append(import_name.replace('.', '/') + '/__init__.py')
        
        # Handle relative imports
        parts = import_name.split('.')
        for i in range(len(parts)):
            partial = '/'.join(parts[:i+1])
            paths.append(f"{partial}.py")
            paths.append(f"{partial}/__init__.py")
        
        return paths
    
    def load_deletion_list(self):
        """Load all files marked for deletion from finalcd2.md analysis"""
        
        # Test scripts to delete
        test_delete = [
            'test_analysis_structure.py',
            'test_extraction.py',
            'test_extraction_fixed.py',
            'test_failfast.py',
            'test_format_defense.py',
            'test_helper_solution.py',
            'test_helper_vs_timeline.py',
            'test_helpers.py',
            'test_which_wrapper.py'
        ]
        
        # Cleanup/analysis scripts
        cleanup_scripts = [
            'analyze_all_deletable_files.py',
            'analyze_all_outputs.py',
            'analyze_data_structures.py',
            'analyze_defensive_programming.py',
            'analyze_test_files.py',
            'check_ml_data_exists.py',
            'count_all_features.py',
            'debug_ml_functions.py',
            'detect_tiktok_creative_elements.py',
            'discovery_analysis.py',
            'final_validation.py',
            'hardcode_settings.py',
            'investigate_helper_intent.py',
            'remove_claude_refs.py',
            'scipy_compat.py',
            'trace_scene_extraction.py',
            'verify_feat_installation.py',
            'smart_cleanup_phase2.py',
            'verify_production_safety.py'  # This script itself
        ]
        
        # Claude-specific files
        claude_files = [
            'rumiai_v2/core/models/prompt.py',
            'rumiai_v2/processors/ml_data_extractor.py',
            'rumiai_v2/processors/prompt_builder.py',
            'rumiai_v2/processors/output_adapter.py',
            'rumiai_v2/api/claude_client.py'
        ]
        
        # All files to delete
        self.files_to_delete = set(test_delete + cleanup_scripts + claude_files)
        
        # Add directories
        self.files_to_delete.add('tests/')
        self.files_to_delete.add('rumiai_v2/prompts/')
        self.files_to_delete.add('prompt_templates/')
        
        print(f"\n📋 Files marked for deletion: {len(self.files_to_delete)}")
        return self.files_to_delete
    
    def check_production_safety(self):
        """Check if any file marked for deletion is used in production"""
        print("\n" + "=" * 70)
        print("SAFETY CHECK: Production vs Deletion List")
        print("=" * 70)
        
        conflicts = []
        for file_to_delete in self.files_to_delete:
            # Check exact match
            if file_to_delete in self.production_files:
                conflicts.append(file_to_delete)
            
            # Check if it's a directory containing production files
            if file_to_delete.endswith('/'):
                for prod_file in self.production_files:
                    if prod_file.startswith(file_to_delete):
                        conflicts.append(f"{file_to_delete} contains {prod_file}")
        
        if conflicts:
            print("\n❌ CRITICAL: Found production files in deletion list!")
            for conflict in conflicts:
                print(f"   - {conflict}")
                self.critical_errors.append(conflict)
        else:
            print("\n✅ SAFE: No production files in deletion list")
        
        return len(conflicts) == 0
    
    def scan_for_claude_js_remnants(self):
        """Scan production files for Claude/JS remnants"""
        print("\n" + "=" * 70)
        print("DEAD CODE SCAN: Claude & JS Remnants")
        print("=" * 70)
        
        patterns = {
            'claude': ['claude', 'Claude', 'CLAUDE', 'anthropic'],
            'prompts': ['prompt', 'Prompt', 'PROMPT'],
            'nodejs': ['node.js', 'Node.js', 'nodejs', 'subprocess', 'legacy_mode'],
            'dead_conditions': ['use_python_only_processing', 'use_ml_precompute']
        }
        
        for prod_file in self.production_files:
            if not os.path.exists(prod_file):
                continue
                
            try:
                with open(prod_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for category, keywords in patterns.items():
                    for keyword in keywords:
                        if keyword.lower() in content.lower():
                            # Find line numbers
                            for i, line in enumerate(lines, 1):
                                if keyword.lower() in line.lower():
                                    # Check if it's a conditional that's always True/False
                                    if 'if' in line and 'use_python_only_processing' in line:
                                        self.claude_js_remnants.append({
                                            'file': prod_file,
                                            'line': i,
                                            'category': 'dead_condition',
                                            'code': line.strip(),
                                            'severity': 'HIGH'
                                        })
                                    elif 'claude' in keyword.lower() and 'claude' in line.lower():
                                        # Skip comments
                                        if not line.strip().startswith('#'):
                                            self.claude_js_remnants.append({
                                                'file': prod_file,
                                                'line': i,
                                                'category': category,
                                                'code': line.strip(),
                                                'severity': 'CRITICAL'
                                            })
            except Exception as e:
                pass
        
        # Print findings
        if self.claude_js_remnants:
            print(f"\n⚠️  Found {len(self.claude_js_remnants)} potential remnants:")
            
            by_file = {}
            for remnant in self.claude_js_remnants:
                if remnant['file'] not in by_file:
                    by_file[remnant['file']] = []
                by_file[remnant['file']].append(remnant)
            
            for file, remnants in by_file.items():
                print(f"\n   {file}:")
                for r in remnants[:3]:  # Show first 3 per file
                    print(f"      Line {r['line']}: {r['code'][:60]}...")
                if len(remnants) > 3:
                    print(f"      ... and {len(remnants)-3} more")
        else:
            print("\n✅ No Claude/JS remnants found in production code")
        
        return len(self.claude_js_remnants) == 0
    
    def verify_ml_services_complete(self):
        """Verify all ML services are properly connected"""
        print("\n" + "=" * 70)
        print("ML SERVICES VERIFICATION")
        print("=" * 70)
        
        required_ml = [
            'YOLOService',
            'WhisperService', 
            'MediaPipeService',
            'OCRService',
            'SceneDetectionService',
            'AudioEnergyService'
        ]
        
        ml_services_file = 'rumiai_v2/api/ml_services.py'
        if os.path.exists(ml_services_file):
            with open(ml_services_file, 'r') as f:
                content = f.read()
            
            found = []
            missing = []
            for service in required_ml:
                if service in content:
                    found.append(service)
                else:
                    missing.append(service)
            
            if missing:
                print(f"⚠️  Missing ML services: {missing}")
                return False
            else:
                print(f"✅ All {len(required_ml)} ML services present")
                return True
        else:
            print("❌ ML services file not found!")
            return False
    
    def generate_report(self):
        """Generate final safety report"""
        print("\n" + "=" * 70)
        print("FINAL SAFETY VERIFICATION REPORT")
        print("=" * 70)
        
        all_safe = True
        
        # 1. Production safety
        prod_safe = len(self.critical_errors) == 0
        if prod_safe:
            print("\n✅ Production Safety: PASSED")
            print("   - No production files in deletion list")
        else:
            print("\n❌ Production Safety: FAILED")
            print(f"   - {len(self.critical_errors)} conflicts found")
            all_safe = False
        
        # 2. Dead code removal
        dead_code_clean = len(self.claude_js_remnants) == 0
        if dead_code_clean:
            print("\n✅ Dead Code Removal: COMPLETE")
            print("   - No Claude/JS remnants in production")
        else:
            print("\n⚠️  Dead Code Removal: INCOMPLETE")
            print(f"   - {len(self.claude_js_remnants)} remnants found")
            all_safe = False
        
        # 3. ML services
        ml_complete = self.verify_ml_services_complete()
        if not ml_complete:
            all_safe = False
        
        # 4. Summary
        print("\n" + "=" * 70)
        if all_safe:
            print("🎉 VERIFICATION PASSED - Safe to proceed with cleanup!")
            print("\nYou can run: python3 smart_cleanup_phase2.py --execute")
        else:
            print("⚠️  VERIFICATION FAILED - Review issues before cleanup")
            print("\nDO NOT proceed with cleanup until issues are resolved")
        
        return all_safe

def main():
    tracer = ProductionFlowTracer()
    
    # 1. Trace production dependencies
    tracer.trace_production_flow()
    
    # 2. Load deletion list
    tracer.load_deletion_list()
    
    # 3. Check safety
    tracer.check_production_safety()
    
    # 4. Scan for remnants
    tracer.scan_for_claude_js_remnants()
    
    # 5. Generate report
    safe = tracer.generate_report()
    
    sys.exit(0 if safe else 1)

if __name__ == "__main__":
    main()