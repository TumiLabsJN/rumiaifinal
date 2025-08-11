#!/usr/bin/env python3
"""
Comprehensive Discovery Analysis for Python-Only Pipeline
Run this ONCE to get all information needed for cleanup decisions.
"""

import os
import sys
import ast
import json
from pathlib import Path
from collections import defaultdict
import builtins

def main():
    print("="*60)
    print("COMPREHENSIVE DISCOVERY ANALYSIS FOR PYTHON-ONLY CLEANUP")
    print("="*60)
    
    # Note: After cleanup, these env vars won't be needed since everything
    # will be hardcoded. Setting them here only for pre-cleanup analysis.
    # os.environ['USE_PYTHON_ONLY_PROCESSING'] = 'true'
    # os.environ['USE_ML_PRECOMPUTE'] = 'true'
    
    results = {}
    
    # 1. CRITICAL PATH ANALYSIS
    print("\n[1/6] Analyzing Critical Path...")
    results['critical_path'] = analyze_critical_path()
    
    # 2. DEPENDENCY MAPPING
    print("\n[2/6] Building Dependency Graph...")
    results['dependencies'] = build_dependency_graph()
    
    # 3. CLAUDE API INVESTIGATION
    print("\n[3/6] Investigating Claude API Usage...")
    results['claude_usage'] = investigate_claude_usage()
    
    # 4. FILE CLASSIFICATION
    print("\n[4/6] Classifying All Files...")
    results['file_classification'] = classify_files()
    
    # 5. RUNTIME VERIFICATION
    print("\n[5/6] Verifying Runtime Independence...")
    results['runtime_check'] = verify_runtime_independence()
    
    # 6. GENERATE FINAL REPORT
    print("\n[6/6] Generating Final Report...")
    generate_report(results)
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("Check 'discovery_report.json' and 'discovery_report.txt' for results")

def analyze_critical_path():
    """Trace what Python-only mode actually uses"""
    imported_modules = []
    original_import = builtins.__import__
    
    def tracking_import(name, *args, **kwargs):
        if 'rumiai' in name or 'scripts' in name:
            imported_modules.append(name)
        return original_import(name, *args, **kwargs)
    
    builtins.__import__ = tracking_import
    
    try:
        from scripts.rumiai_runner import RumiAIRunner
        result = {
            'success': True,
            'modules_imported': list(set(imported_modules)),
            'count': len(set(imported_modules))
        }
    except Exception as e:
        result = {'success': False, 'error': str(e)}
    finally:
        # CRITICAL: Restore original import to avoid conflicts
        builtins.__import__ = original_import
    
    return result

def build_dependency_graph():
    """Build complete dependency graph"""
    def get_imports(file_path):
        imports = []
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
        except:
            pass
        return imports
    
    dep_graph = {}
    claude_imports = []
    
    for py_file in Path('.').rglob('*.py'):
        if 'venv' not in str(py_file) and '__pycache__' not in str(py_file):
            imports = get_imports(py_file)
            dep_graph[str(py_file)] = imports
            
            # Track Claude imports
            for imp in imports:
                if 'claude' in imp.lower():
                    claude_imports.append((str(py_file), imp))
    
    return {
        'total_files': len(dep_graph),
        'claude_imports': claude_imports[:10],  # First 10
        'sample_graph': dict(list(dep_graph.items())[:5])  # Sample
    }

def investigate_claude_usage():
    """Check if Claude is bypassed in Python-only mode"""
    # Check key file for bypass logic
    try:
        with open('scripts/rumiai_runner.py', 'r') as f:
            lines = f.readlines()
        
        # Find Python-only bypass
        bypass_line = None
        for i, line in enumerate(lines, 1):
            if 'use_python_only_processing' in line and 'if' in line:
                bypass_line = i
                break
        
        # Find zero cost evidence
        zero_cost_line = None
        for i, line in enumerate(lines, 1):
            if 'estimated_cost=0' in line or 'estimated_cost = 0' in line:
                zero_cost_line = i
                break
        
        return {
            'bypass_found': bypass_line is not None,
            'bypass_line': bypass_line,
            'zero_cost_found': zero_cost_line is not None,
            'zero_cost_line': zero_cost_line
        }
    except:
        return {'error': 'Could not analyze rumiai_runner.py'}

def classify_files():
    """Classify all files for deletion"""
    categories = defaultdict(list)
    
    # JavaScript files
    for f in Path('.').glob('**/*.js'):
        if 'venv' not in str(f) and 'node_modules' not in str(f):
            if 'whisper.cpp' in str(f):
                categories['KEEP_WHISPER'].append(str(f))
            else:
                categories['DELETE_JAVASCRIPT'].append(str(f))
    
    # Package files
    for f in Path('.').glob('**/package*.json'):
        if 'venv' not in str(f) and 'whisper.cpp' not in str(f):
            categories['DELETE_JAVASCRIPT'].append(str(f))
    
    # Claude files
    for f in Path('.').rglob('*claude*'):
        if f.is_file() and 'venv' not in str(f) and '__pycache__' not in str(f):
            categories['DELETE_CLAUDE'].append(str(f))
    
    # Prompt files
    for f in Path('prompt_templates').glob('*.txt'):
        categories['DELETE_PROMPTS'].append(str(f))
    
    # Legacy files
    for f in Path('.').rglob('*adapter*.py'):
        if 'venv' not in str(f) and '__pycache__' not in str(f):
            categories['DELETE_LEGACY'].append(str(f))
    
    # ML/Precompute files (KEEP)
    for f in Path('.').rglob('*precompute*.py'):
        if 'venv' not in str(f) and '__pycache__' not in str(f):
            categories['KEEP_PYTHON_ONLY'].append(str(f))
    
    return {
        'to_delete': {
            'javascript': categories['DELETE_JAVASCRIPT'],
            'claude': categories['DELETE_CLAUDE'],
            'prompts': categories['DELETE_PROMPTS'],
            'legacy': categories['DELETE_LEGACY']
        },
        'to_keep': {
            'whisper': categories['KEEP_WHISPER'],
            'python_only': categories['KEEP_PYTHON_ONLY']
        },
        'total_to_delete': sum(len(v) for k, v in categories.items() if 'DELETE' in k)
    }

def verify_runtime_independence():
    """Verify Python-only mode works independently"""
    claude_accessed = []
    original_import = builtins.__import__
    
    def blocking_import(name, *args, **kwargs):
        if 'claude' in name.lower() and 'precompute' not in name:
            claude_accessed.append(name)
        return original_import(name, *args, **kwargs)
    
    builtins.__import__ = blocking_import
    
    try:
        from scripts.rumiai_runner import RumiAIRunner
        from rumiai_v2.processors.precompute_functions import COMPUTE_FUNCTIONS
        
        result = {
            'can_import': True,
            'claude_modules_imported': claude_accessed,
            'compute_functions_available': list(COMPUTE_FUNCTIONS.keys()),
            'needs_refactoring': len(claude_accessed) > 0
        }
    except Exception as e:
        result = {'can_import': False, 'error': str(e)}
    finally:
        # CRITICAL: Restore original import to avoid system corruption
        builtins.__import__ = original_import
    
    return result

def generate_report(results):
    """Generate comprehensive report with error handling"""
    try:
        # Save JSON report
        with open('discovery_report.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ Failed to write JSON report: {e}")
        # Continue with text report even if JSON fails
    
    # Generate text report
    try:
        with open('discovery_report.txt', 'w') as f:
            f.write("DISCOVERY ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Critical findings
            f.write("CRITICAL FINDINGS:\n")
            if results['runtime_check'].get('needs_refactoring'):
                f.write("⚠️  REFACTORING REQUIRED: Claude modules still imported\n")
                f.write(f"   Modules: {results['runtime_check']['claude_modules_imported']}\n")
            else:
                f.write("✅ No refactoring needed - Python-only is independent\n")
            
            f.write(f"\n✅ Python-only bypass found at line: {results['claude_usage'].get('bypass_line')}\n")
            f.write(f"✅ Zero cost confirmed at line: {results['claude_usage'].get('zero_cost_line')}\n")
            f.write(f"✅ Compute functions available: {len(results['runtime_check'].get('compute_functions_available', []))}\n")
            
            # Files to delete
            f.write(f"\nFILES TO DELETE: {results['file_classification']['total_to_delete']} total\n")
            for category, files in results['file_classification']['to_delete'].items():
                f.write(f"\n{category.upper()}: {len(files)} files\n")
                for file in files[:5]:  # First 5
                    f.write(f"  - {file}\n")
                if len(files) > 5:
                    f.write(f"  ... and {len(files)-5} more\n")
            
            # Files to keep
            f.write(f"\nFILES TO KEEP:\n")
            for category, files in results['file_classification']['to_keep'].items():
                f.write(f"\n{category.upper()}: {len(files)} files\n")
                for file in files[:3]:  # First 3
                    f.write(f"  - {file}\n")
    except Exception as e:
        print(f"⚠️ Failed to write text report: {e}")
    
    # Print summary regardless of file write success
    try:
        print("\n" + "="*60)
        print("SUMMARY:")
        print(f"📊 Total files to delete: {results['file_classification']['total_to_delete']}")
        print(f"⚠️  Needs refactoring: {results['runtime_check'].get('needs_refactoring', False)}")
        print(f"✅ Compute functions: {len(results['runtime_check'].get('compute_functions_available', []))}")
        print("="*60)
    except Exception as e:
        print(f"❌ Failed to print summary: {e}")

if __name__ == "__main__":
    main()