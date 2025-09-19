# Adding New Analysis Types to RumiAI - Critical Checklist

**Version**: 1.0.0  
**Last Updated**: 2025-01-11  
**Purpose**: Step-by-step guide to properly integrate new analysis types without breaking existing architecture

## Executive Summary

Adding a new analysis type to RumiAI requires careful coordination across multiple files and systems. This guide captures lessons learned from adding the temporal markers analysis type, including common pitfalls and critical integration points.

## Critical Files to Update (8 Total)

When adding a new analysis type, you MUST update ALL of these files:

### 1. Core Implementation Files
- [ ] `rumiai_v2/processors/[new_analysis].py` - Create the analysis module
- [ ] `rumiai_v2/processors/precompute_functions.py` - Register the compute function
- [ ] `scripts/rumiai_runner.py` - Add to processing pipeline

### 2. Documentation Files (Often Forgotten!)
- [ ] `ML_FEATURES_DOCUMENTATION_V2.md` - Add new section with features
- [ ] `FlowStructure.md` - Update analysis count and add to table
- [ ] `Codemappingfinal.md` - Add to Python Compute Functions table
- [ ] `rumiai_v2/config/settings.py` - Enable in precompute settings

### 3. Data Flow Integration
- [ ] `rumiai_v2/processors/video_analyzer.py` - Include in analysis pipeline

## Step-by-Step Implementation Guide

### Step 1: Create the Analysis Module

Create `rumiai_v2/processors/[new_analysis].py`:

```python
def generate_[analysis_name](timelines: dict, duration: float) -> dict:
    """
    Generate [analysis] markers for video engagement analysis.
    
    Args:
        timelines: ML timeline data from video analysis
        duration: Video duration in seconds
        
    Returns:
        dict: Analysis results in professional format
    """
    # Implementation here
    return analysis_results
```

### Step 2: Register in Precompute Functions

Update `rumiai_v2/processors/precompute_functions.py`:

```python
# Add import
from rumiai_v2.processors.[new_analysis] import generate_[analysis_name]

# Add to COMPUTE_FUNCTIONS dict
COMPUTE_FUNCTIONS = {
    # ... existing entries ...
    '[analysis_name]': generate_[analysis_name],
}
```

### Step 3: Add to Runner Pipeline

Update `scripts/rumiai_runner.py`:

#### 3a. Add to ANALYSIS_TYPES list:
```python
ANALYSIS_TYPES = [
    'creative_density',
    'emotional_journey', 
    'person_framing',
    'scene_pacing',
    'speech_analysis',
    'visual_overlay_analysis',
    'metadata_analysis',
    '[new_analysis]'  # ADD HERE
]
```

#### 3b. Implement backward compatibility (3-file output):
```python
def save_analysis_result(self, video_id: str, analysis_type: str, data: dict) -> Path:
    """Save analysis result in 3-file backward compatible format."""
    # CRITICAL: Must save 3 files for legacy system compatibility
    # 1. complete file (full response)
    # 2. ml file (unprefixed format)
    # 3. result file (prefixed format)
```

### Step 4: Update Documentation

#### 4a. ML_FEATURES_DOCUMENTATION_V2.md
- Update total count from 7 to 8 analysis types
- Add new section (e.g., "## 8. [Analysis Name] Features (Python)")
- Include ~20 features with descriptions
- Update Table of Contents

#### 4b. FlowStructure.md
- Change "7 Python Analysis Types" to "8 Python Analysis Types"
- Add row to analysis table with function name
- Add description in Analysis Descriptions section

#### 4c. Codemappingfinal.md
- Add row to Python Compute Functions table
- Include function name, output format, and features

### Step 5: Enable in Settings

Update `rumiai_v2/config/settings.py`:

```python
self.precompute_enabled_prompts = {
    # ... existing entries ...
    '[analysis_name]': True  # ADD HERE
}
```

## Common Bugs and How to Avoid Them

### Bug 1: Analysis Not Executing
**Symptom**: New analysis type doesn't run  
**Cause**: Not added to ANALYSIS_TYPES in rumiai_runner.py  
**Fix**: Ensure analysis is in the ANALYSIS_TYPES list

### Bug 2: Import Errors
**Symptom**: "No module named" or "cannot import"  
**Cause**: Missing import in precompute_functions.py  
**Fix**: Add proper import statement and register in COMPUTE_FUNCTIONS

### Bug 3: Backward Compatibility Broken
**Symptom**: Legacy systems can't read output  
**Cause**: Not saving 3-file structure per insight  
**Fix**: Implement save_analysis_result() with complete/ml/result files

### Bug 4: Wrong Output Location
**Symptom**: Files saved in wrong directory  
**Cause**: Hardcoded paths or incorrect directory structure  
**Fix**: Use `insights/{video_id}/{analysis_type}/` pattern

### Bug 5: Documentation Inconsistency
**Symptom**: Documentation shows wrong count of analysis types  
**Cause**: Forgot to update all 3 documentation files  
**Fix**: Update ML_FEATURES, FlowStructure, and Codemappingfinal

### Bug 6: Missing Prefix Mapping
**Symptom**: Result file has wrong format  
**Cause**: No prefix mapping for new analysis type  
**Fix**: Add to get_prefix_for_type() in rumiai_runner.py

## Format Requirements

### 1. File Naming Convention
```
insights/{video_id}/{analysis_type}/{analysis_type}_{suffix}_{timestamp}.json
```
Where suffix is: `complete`, `ml`, or `result`

### 2. Output Structure (6-Block Format)
```json
{
  "[prefix]CoreMetrics": {},
  "[prefix]Dynamics": {},
  "[prefix]Interactions": {},
  "[prefix]KeyEvents": {},
  "[prefix]Patterns": {},
  "[prefix]Quality": {}
}
```

### 3. Prefix Conversion Rules
- ML format: No prefix (e.g., `CoreMetrics`)
- RESULT format: With prefix (e.g., `densityCoreMetrics`)
- Complete format: Contains full response with metadata

## Integration Testing Checklist

After adding a new analysis type, verify:

- [ ] Analysis runs without errors
- [ ] Saves 3 files per insight (complete, ml, result)
- [ ] Files are in correct directory structure
- [ ] Prefix conversion works correctly
- [ ] Documentation reflects new count (8 instead of 7)
- [ ] Function is registered in precompute_functions.py
- [ ] Settings enable the new analysis
- [ ] Backward compatibility maintained

## Processing Flow

### URL Processing (Only Available Flow)
```bash
python3 scripts/rumiai_runner.py "https://tiktok.com/..."
```
- Downloads video via Apify
- Runs full ML pipeline
- Generates all 8 analyses
- Saves to insights/{video_id}/ with 3-file structure

## Critical Lessons Learned

### 1. Always Update Documentation
The most common mistake is forgetting to update documentation files. They're not just reference - they're actively used by the system and other developers.

### 2. Maintain Backward Compatibility
Legacy systems depend on the 3-file structure. Never simplify to single file output without careful migration planning.

### 3. Test Both Processing Flows
New analysis must work in both URL processing and Video ID reprocessing flows.

### 4. Use Consistent Naming
- Function: `generate_[analysis]()` or `compute_[analysis]_wrapper()`
- Directory: `insights/{video_id}/[analysis_type]/`
- Prefix: Define in get_prefix_for_type()

### 5. Check Import Paths
Always use absolute imports, not relative:
```python
# Good
from rumiai_v2.processors.temporal_markers import generate_markers

# Bad
from .temporal_markers import generate_markers
```

## Quick Reference Commands

### Test New Analysis Type
```bash
# Test with URL (full pipeline)
python3 scripts/rumiai_runner.py "https://www.tiktok.com/@user/video/123"

# Check output structure
ls -la insights/123456789/[new_analysis]/
```

### Verify File Count
```bash
# Should show 3 files per analysis
find insights/*/[new_analysis]/ -name "*.json" | wc -l
```

## Troubleshooting Guide

### Analysis Not Running?
1. Check ANALYSIS_TYPES list in rumiai_runner.py
2. Verify import in precompute_functions.py
3. Confirm settings.py enables the analysis
4. Check for Python syntax errors in analysis module

### Wrong Output Format?
1. Verify prefix mapping in get_prefix_for_type()
2. Check convert_to_ml_format() logic
3. Ensure 6-block structure in analysis module
4. Validate JSON structure

### Import Errors?
1. Use absolute imports
2. Check module is in correct directory
3. Verify __init__.py files if needed
4. Test import in Python REPL

## Final Checklist

Before considering a new analysis type complete:

- [ ] All 8 files updated (see Critical Files list)
- [ ] Both processing flows tested
- [ ] 3-file backward compatibility verified
- [ ] Documentation shows correct count
- [ ] No import errors
- [ ] Output in correct directory
- [ ] Prefix conversion working
- [ ] JSON structure validated
- [ ] Integration with existing analyses verified
- [ ] Error handling implemented

## Contact for Issues

If you encounter issues not covered in this guide:
1. Check existing analysis implementations for reference
2. Verify all file updates against this checklist
3. Test with minimal video first
4. Check logs for specific error messages

Remember: Adding a new analysis type affects the entire pipeline. Take time to verify each integration point carefully.