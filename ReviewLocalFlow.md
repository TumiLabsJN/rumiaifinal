# Local Video Testing Flow - Maintenance Guide

## Overview
The local video testing infrastructure (`scripts/local_video_runner.py`) allows testing RumiAI's video analysis pipeline without TikTok uploads. This guide explains when and how to maintain synchronization with production code.

## 🔄 When to Update Local Analysis Code

### Immediate Updates Required
Update `local_video_runner.py` when production changes:

1. **Field Naming Conventions**
   - New analysis types added to `COMPUTE_FUNCTIONS`
   - Changes to 6-block structure (CoreMetrics, Dynamics, etc.)
   - Modified prefixed naming patterns (e.g., `densityCoreMetrics` → `densityCore`)

2. **Output Structure**
   - Number of output files changes (currently 3: complete, ml, result)
   - File naming conventions change
   - Directory structure modifications

3. **Professional Wrappers**
   - New analyses using `precompute_professional_wrappers.py`
   - Changes to camelCase patterns (personFraming, scenePacing, etc.)

### No Updates Needed
These production changes propagate automatically:

- ML model updates (YOLO, Whisper, OCR)
- Timeline builder logic changes
- Temporal marker processing updates
- Core analysis algorithm changes

## 🛠️ How to Update

### Step 1: Run Synchronization Check
```bash
python3 scripts/verify_sync.py
```

This detects:
- New/removed analysis types
- Unmapped field patterns
- Output structure mismatches
- Professional wrapper changes

### Step 2: Update Field Mappings
If sync check shows unmapped patterns, edit `scripts/local_video_runner.py`:

```python
# In convert_to_ml_format() method, add new mappings:
key_mappings = {
    # Add new analysis type
    'newAnalysisCoreMetrics': 'CoreMetrics',
    'newAnalysisDynamics': 'Dynamics',
    # ... etc
}
```

### Step 3: Verify Fix
```bash
# Re-run sync check
python3 scripts/verify_sync.py

# Test with actual video
python3 scripts/local_video_runner.py test_videos/video_01_highenergy_peaks.mp4
```

## 🔍 Granular Verification

### When to Use `compare_ml_results.py`

Run detailed comparisons when:

1. **Testing Production Changes**
   ```bash
   # Process same video through both pipelines
   python3 scripts/compare_ml_results.py <prod_video_id> <test_video_id> <analysis_type>
   ```

2. **Validating ML Consistency**
   - After ML model updates
   - When detection counts seem off
   - To establish acceptable variance ranges

3. **Debugging Specific Issues**
   ```bash
   # Compare specific analysis
   python3 scripts/compare_ml_results.py 7518107389090876727 TestVideo creative_density
   ```

### Expected Variations
Normal differences between production and test:

| Analysis Type | Acceptable Variance | Common Causes |
|--------------|-------------------|---------------|
| creative_density | ±5-15% elements | YOLO detection variance |
| emotional_journey | ±10% intensity | Emotion model confidence |
| metadata_analysis | 100% hashtags | Mock vs real metadata |
| person_framing | ±5% visibility | Face detection threshold |
| scene_pacing | ±1-2 scenes | Scene detection sensitivity |
| speech_analysis | ±1% words | Whisper transcription |
| visual_overlay | ±10-15% overlays | OCR sensitivity |
| temporal_markers | ±2s timing | Frame extraction timing |

### Red Flags 🚩
Investigate if you see:
- >20% variance in any metric
- Complete detection failure (0 elements when expecting many)
- Structural differences (missing fields)
- File generation failures

## 📋 Maintenance Checklist

### After Production Code Changes
```bash
# 1. Check synchronization
python3 scripts/verify_sync.py

# 2. If issues found, update local_video_runner.py
#    - Add new field mappings
#    - Update analysis type list
#    - Modify save_results if needed

# 3. Test with known video
python3 scripts/local_video_runner.py test_videos/video_01_highenergy_peaks.mp4

# 4. Compare outputs
python3 scripts/check_all_formats.py  # Structure check
python3 scripts/compare_all_values.py  # Value comparison

# 5. Document any new variance patterns
```

### Weekly Health Check
```bash
# Process test video and verify outputs
python3 scripts/local_video_runner.py test_videos/video_01_highenergy_peaks.mp4

# Check latest production/test alignment
ls -la insights/*/creative_density/ | head -20

# Run sync verification
python3 scripts/verify_sync.py
```

## 🎯 Key Files Reference

### Core Scripts
- `scripts/local_video_runner.py` - Main local testing script
- `scripts/verify_sync.py` - Production/test synchronization checker
- `scripts/compare_ml_results.py` - Detailed value comparison
- `scripts/compare_outputs.py` - Structure comparison
- `scripts/check_all_formats.py` - Format consistency checker

### Critical Methods
1. **`convert_to_ml_format()`** - Converts prefixed names to generic
   - Location: `local_video_runner.py:213`
   - Most common maintenance point
   - Maps all field name patterns

2. **`save_results()`** - Creates 3 output files per analysis
   - Location: `local_video_runner.py:310`
   - Update if file structure changes

3. **`run_precompute_analysis()`** - Executes all analyses
   - Location: `local_video_runner.py:187`
   - Update if COMPUTE_FUNCTIONS changes

## ⚠️ Common Issues & Solutions

### Issue: "ML file structure differs"
**Cause**: Field names not being converted properly
**Solution**: Add missing mappings to `convert_to_ml_format()`

### Issue: Large value differences (>20%)
**Cause**: Different video files or corrupted video
**Solution**: 
1. Verify same source video
2. Check video integrity with `ffprobe`
3. Re-download if needed

### Issue: Missing analysis outputs
**Cause**: New analysis type not handled
**Solution**: 
1. Run `verify_sync.py`
2. Add new analysis to mappings
3. Test with local video

### Issue: Import errors
**Cause**: Production module structure changed
**Solution**: 
1. Check production imports
2. Update import statements
3. Verify with `python3 -c "from rumiai_v2.processors import COMPUTE_FUNCTIONS"`

## 📊 Testing Workflow

### For New Session
```bash
# 1. Verify environment
python3 scripts/verify_sync.py

# 2. Run test video
python3 scripts/local_video_runner.py test_videos/video_01_highenergy_peaks.mp4

# 3. Check outputs
ls -la insights/video_01_highenergy_peaks/

# 4. If comparing with production
python3 scripts/compare_ml_results.py <prod_id> video_01_highenergy_peaks creative_density
```

### For Production Debugging
```bash
# 1. Get problem video
# Download from TikTok or use provided file

# 2. Process locally
python3 scripts/local_video_runner.py problem_video.mp4

# 3. Compare with production
python3 scripts/compare_ml_results.py <prod_video_id> <local_video_id> <problem_analysis>

# 4. Check for anomalies in specific analysis
cat insights/<video_id>/<analysis>/*_ml_*.json | python3 -m json.tool | less
```

## 🔮 Future Considerations

### Potential Production Changes
Watch for these changes that would require updates:

1. **New ML Services**
   - Would appear in `ml_results` dictionary
   - May need new timeline entry handling

2. **Modified Precompute Returns**
   - Could change field structure
   - Would need mapping updates

3. **New File Outputs**
   - Additional files beyond complete/ml/result
   - Would need `save_results()` update

4. **API Response Format Changes**
   - Could affect complete file structure
   - May need new conversion logic

### Automation Opportunities
Consider automating:
- Daily sync checks via cron
- Automatic mapping generation from production outputs
- Golden dataset validation on schedule

## 📝 Notes for Fresh Session

When starting work in a new session:

1. **Check Recent Changes**
   ```bash
   # See what's new in production
   ls -lt rumiai_v2/processors/precompute*.py | head -5
   
   # Check recent test outputs
   ls -lt insights/*/creative_density/ | head -10
   ```

2. **Verify Test Data**
   ```bash
   # Ensure test videos exist
   ls -la test_videos/*.mp4
   ```

3. **Run Health Check**
   ```bash
   python3 scripts/verify_sync.py
   ```

4. **Remember Key Points**
   - Production uses GENERIC names in ML files
   - Test script must convert prefixed → generic
   - 5-15% ML variance is normal
   - Mock metadata differs from real TikTok data

## 📞 Quick Reference

```bash
# Test local video
python3 scripts/local_video_runner.py <video_path>

# Check sync
python3 scripts/verify_sync.py

# Compare two videos
python3 scripts/compare_ml_results.py <video1> <video2> <analysis>

# Check all formats
python3 scripts/check_all_formats.py

# Compare all values
python3 scripts/compare_all_values.py
```

---

Last Updated: 2025-08-12
Maintained by: RumiAI Local Testing Infrastructure