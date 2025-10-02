# Revert Scene Changes Implementation

## Current State (October 2, 2025)

**Problem**: Scene detection completely broken after implementing scene_changes fix
- **Before**: segment_1 showed 4 scenes (should be 3) - mostly working
- **After**: ALL segments show scene_count = 1 - completely broken

## Files Modified During Implementation

### 1. `/rumiai_v2/api/ml_services_unified.py`

**Lines Added Around 1009-1014:**
```python
# Create scene_change entry for timeline_builder
scene_changes.append({
    'timestamp': start.get_seconds(),
    'scene_number': i + 1,
    'transition_type': 'cut'
})
```

**Return Statement Modified:**
```python
# ADDED scene_changes to return dict
return {
    'scenes': scene_list,
    'scene_changes': scene_changes,  # <-- THIS WAS ADDED
    'total_scenes': len(scene_list),
    'metadata': {'processed': True}
}
```

**Error Case Modified:**
```python
# ADDED scene_changes to error return
return {
    'scenes': [],
    'scene_changes': [],  # <-- THIS WAS ADDED
    'total_scenes': 0,
    'metadata': {'processed': False, 'error': str(e)}
}
```

### 2. `/rumiai_v2/processors/timeline_builder.py`

**Debug Logging Added (Lines ~331-355):**
```python
# All the 🔍 SCENE DEBUG logging statements
logger.info(f"🔍 SCENE DEBUG: Processing {len(scene_changes_list)} scene_changes from scene_data")
logger.info(f"🔍 SCENE DEBUG: scene_data keys: {list(scene_data.keys())}")
# ... multiple debug lines
```

**Main Function Debug Logging Added (Lines ~33-38):**
```python
logger.info(f"🔍 ML_RESULTS DEBUG: Available ML results: {list(ml_results.keys())}")
for model_name, result in ml_results.items():
    logger.info(f"🔍 ML_RESULTS DEBUG: {model_name} - success: {result.success}")
    if model_name == 'scene_detection' and result.success:
        logger.info(f"🔍 ML_RESULTS DEBUG: scene_detection data keys: {list(result.data.keys())}")
```

### 3. `/rumiai_v2/processors/temporal_compute.py`

**Debug Logging Added (Lines ~2010, ~2024-2029):**
```python
logger.info(f"🔍 TEMPORAL DEBUG: Found {len(all_scenes)} scenes in scene_change_timeline for segment {start}s-{end}s")

# And later:
logger.info(f"🔍 TEMPORAL DEBUG: Scene {i+1} ({scene_start}s-{scene_end}s) overlaps with segment {start}s-{end}s")
logger.info(f"🔍 TEMPORAL DEBUG: No scenes found, defaulting to scene_count=1 for segment {start}s-{end}s")
logger.info(f"🔍 TEMPORAL DEBUG: Final scene_count for segment {start}s-{end}s: {scene_count}")
```

### 4. Threshold Changes (Already Reverted)
- Extended thresholds from [35.0, 30.0, 25.0, 20.0] to include 15.0, 12.0, 10.0
- Added micro-scene filtering logic
- **Status**: Already reverted to original

## Complete Revert Instructions

### Step 1: Revert ml_services_unified.py

**Remove scene_changes creation and return:**
```python
# REMOVE these lines from the scene processing loop:
# Create scene_change entry for timeline_builder
scene_changes.append({
    'timestamp': start.get_seconds(),
    'scene_number': i + 1,
    'transition_type': 'cut'
})

# CHANGE return statement back to original:
return {
    'scenes': scene_list,
    # 'scene_changes': scene_changes,  # <-- REMOVE THIS LINE
    'total_scenes': len(scene_list),
    'metadata': {'processed': True}
}

# CHANGE error return back to original:
return {
    'scenes': [],
    # 'scene_changes': [],  # <-- REMOVE THIS LINE
    'total_scenes': 0,
    'metadata': {'processed': False, 'error': str(e)}
}

# REMOVE scene_changes list initialization:
# scene_changes = []  # <-- REMOVE THIS LINE
```

### Step 2: Remove Debug Logging from timeline_builder.py

**Remove all debug logging lines containing:**
- `🔍 SCENE DEBUG:`
- `🔍 ML_RESULTS DEBUG:`

**Specifically remove:**
```python
# Remove these lines from _add_scene_entries method:
scene_changes_list = scene_data.get('scene_changes', [])
logger.info(f"🔍 SCENE DEBUG: Processing {len(scene_changes_list)} scene_changes from scene_data")
logger.info(f"🔍 SCENE DEBUG: scene_data keys: {list(scene_data.keys())}")

for i, scene_change in enumerate(scene_changes_list):
    logger.info(f"🔍 SCENE DEBUG: Processing scene_change {i+1}: {scene_change}")
    # ... and all other debug lines

# Change back to original:
for scene_change in scene_data.get('scene_changes', []):
```

**Remove from build_timeline method:**
```python
# Remove these lines:
logger.info(f"🔍 ML_RESULTS DEBUG: Available ML results: {list(ml_results.keys())}")
for model_name, result in ml_results.items():
    logger.info(f"🔍 ML_RESULTS DEBUG: {model_name} - success: {result.success}")
    if model_name == 'scene_detection' and result.success:
        logger.info(f"🔍 ML_RESULTS DEBUG: scene_detection data keys: {list(result.data.keys())}")
```

### Step 3: Remove Debug Logging from temporal_compute.py

**Remove all debug logging lines containing:**
- `🔍 TEMPORAL DEBUG:`

**Specifically:**
```python
# Remove:
logger.info(f"🔍 TEMPORAL DEBUG: Found {len(all_scenes)} scenes in scene_change_timeline for segment {start}s-{end}s")
logger.info(f"🔍 TEMPORAL DEBUG: Scene {i+1} ({scene_start}s-{scene_end}s) overlaps with segment {start}s-{end}s")
logger.info(f"🔍 TEMPORAL DEBUG: No scenes found, defaulting to scene_count=1 for segment {start}s-{end}s")
logger.info(f"🔍 TEMPORAL DEBUG: Final scene_count for segment {start}s-{end}s: {scene_count}")
```

### Step 4: Clear All Caches

```bash
# Clear scene detection cache
rm -rf /home/jorge/rumiaifinal/scene_detection_outputs/

# Clear temporal windows cache
rm -f /home/jorge/rumiaifinal/insights/*_temporal_windows_updated.json

# Clear any other relevant caches
```

### Step 5: Test Original State

**Expected Result After Revert:**
- Video 7480428850522950920 should return to original behavior
- segment_1 should show 4 scenes (the original "wrong" count)
- Other segments should work correctly
- This confirms we're back to the pre-implementation state

## Verification Steps

1. **Run test on video 7480428850522950920**
2. **Check segment_1 scene_count = 4** (original wrong value)
3. **Check other segments have proper scene counts** (not all 1s)
4. **Confirm we're back to "mostly working" state**

## Alternative: Partial Revert

If you want to keep the data structure fix but remove debug logging:
- **Keep**: scene_changes array creation and return in ml_services_unified.py
- **Remove**: All debug logging from timeline_builder.py and temporal_compute.py
- **Investigate**: Why scene_changes array is empty in timeline_builder

## Risk Assessment

**Revert Risk**: Low - returns to known working state
**Time Required**: 15-30 minutes
**Testing**: Single video test sufficient to verify revert success

---

**Recommendation**: Complete revert to understand what broke, then re-implement with minimal changes and proper testing at each step.