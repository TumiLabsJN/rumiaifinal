# Scene Pacing Analysis Architecture

## Overview
The scene_pacing analysis flow detects scene changes and analyzes video editing rhythm, but contains significant Claude API optimization code despite never using Claude. The system uses PySceneDetect for scene detection and complex metrics computation with multiple terminology inconsistencies.

## Current Architecture Flow

### 1. Scene Detection Service
```
video_analyzer.py → _run_scene_detection() → ml_services.run_scene_detection()
                                            ↓
                            PySceneDetect with adaptive thresholds
                                            ↓
                        scenes: [{start_time: 0.0, end_time: 1.17}, ...]
```

**Implementation Details** (ml_services.py:111-161):
- Uses PySceneDetect's ContentDetector
- Adaptive threshold selection: [20.0, 15.0, 10.0]
- Target: 1-5 second average scene length
- Output: Numeric timestamps (float seconds)

### 2. Timeline Builder Integration
```python
# timeline_builder.py:323-376
def _add_scene_entries(self, ml_results):
    scenes = ml_results['scene_detection'].data.get('scenes', [])
    for scene in scenes:
        entry = TimelineEntry(
            start=scene['start_time'],  # Numeric: 1.1666667
            end=None,                    # Instantaneous
            entry_type='scene_change',
            data={'transition_type': 'cut', 'scene_index': i}
        )
```

### 3. Timeline Extraction & Format Conversion
```python
# precompute_functions.py:420-509
# CRITICAL BUG-PRONE AREA: Multiple timestamp format handling
def _extract_timelines_from_analysis(analysis_dict):
    # Problem: Must handle both formats
    # - String: "1.5s" (from some sources)
    # - Numeric: 1.1666667 (from ML detection)
    
    # Complex conversion logic for sceneChangeTimeline
    for entry in timeline_entries:
        if entry.get('entry_type') == 'scene_change':
            start_value = entry.get('start', 0)
            
            # Format detection and conversion
            if isinstance(start_value, str) and start_value.endswith('s'):
                start_seconds = float(start_value[:-1])
            elif isinstance(start_value, (int, float)):
                start_seconds = float(start_value)
            
            # Create timeline key: "0-1s", "1-2s", etc.
            timestamp = f"{int(start_seconds)}-{int(start_seconds)+1}s"
```

**Over-Engineering Issue**: Handles up to 1000 scene changes per second
```python
# Lines 493-509: Fractional second handling
for j, change in enumerate(changes):
    fraction = j / 1000.0  # Supports 1000 changes/second
    timestamp = f"{start_time:.3f}-{end_time:.3f}s"
    # Creates: "0.000-0.001s", "0.001-0.002s", etc.
```

### 4. Scene Pacing Metrics Computation
```python
# precompute_functions_full.py:2650-3087
def compute_scene_pacing_metrics(scene_timeline, video_duration, ...):
    # TERMINOLOGY CONFUSION: Uses "shots" internally
    total_shots = len(scene_times)  # Actually scenes
    avg_shot_duration = ...          # Actually scene duration
    shots_per_minute = ...           # Actually scenes per minute
    
    # Claude API Optimization (UNUSED)
    # Lines 4480-4517: Payload size optimization for Claude
    payload_size = len(json.dumps(context_data))
    if payload_size > 1_000_000:
        print("⚠️ Large payload warning")
    
    # Complex metrics computation
    return {
        'total_shots': total_shots,  # Wrong terminology
        'avg_shot_duration': avg_shot_duration,
        'pacing_classification': 'fast',
        'rhythm_consistency': 'varied',
        # ... 50+ other metrics
    }
```

### 5. Professional Wrapper Conversion
```python
# precompute_professional_wrappers.py:183-233
def convert_to_scene_pacing_professional(basic_metrics):
    # CRITICAL: Terminology mapping required
    return {
        "scenePacingCoreMetrics": {
            # Map "shots" → "scenes"
            "totalScenes": basic_metrics.get('total_shots'),  # FIX
            "averageSceneDuration": basic_metrics.get('avg_shot_duration'),
            "scenesPerMinute": basic_metrics.get('shots_per_minute'),
            # ... defaults for missing fields
        }
    }
```

### 6. Output Format Structure
```json
{
  "scenePacingCoreMetrics": {
    "totalScenes": 7,
    "averageSceneDuration": 2.0,
    "scenesPerMinute": 30.0,
    "rhythmConsistency": "varied"
  },
  "scenePacingDynamics": {...},
  "scenePacingInteractions": {...},
  "scenePacingKeyEvents": {...},
  "scenePacingPatterns": {...},
  "scenePacingQuality": {...}
}
```

## Identified Issues

### 1. **Claude API Optimization for Unused System**
```python
# precompute_functions_full.py:4480-4517
# Entire block optimizes payload for Claude API
result = runner.run_claude_prompt(...)  # Claude never actually called
```
- Payload size checks
- Compression logic
- API limit handling
- All unnecessary since using Python-only computation

### 2. **Terminology Confusion: Shots vs Scenes**
- ML detection: "scenes"
- Core computation: "shots" (everywhere in code)
- Professional output: "scenes"
- Requires mapping layer: `total_shots` → `totalScenes`

### 3. **Timestamp Format Conversion Overhead**
```
Flow: 1.1666667 → "1-2s" → parse to 1 → display
```
- Multiple conversions for same data
- String manipulation overhead
- Loss of precision

### 4. **Over-Engineered Edge Cases**
- Supports 1000+ scene changes per second (never happens)
- Complex fractional timestamp generation
- FPS context manager imported but unused

### 5. **Dual Timeline Structures**
- `sceneChangeTimeline`: For computation
- `sceneTimeline`: For segments (rarely used)
- Redundant data storage

### 6. **Professional Wrapper Defaults**
Many fields have hardcoded defaults:
```python
"transitionSmoothing": 0.8,  # Always 0.8
"audioVisualSync": 0.8,      # Always 0.8
"narrativeFlow": 0.75,       # Always 0.75
```

## Safe Optimization Strategy

### Phase 1: Remove Claude API References (Zero Risk)
**Goal**: Clean up unused Claude optimization code

```python
# DELETE from precompute_functions_full.py:4480-4517
# Remove entire Claude prompt runner block
# This code is never executed in Python-only mode

# KEEP the actual computation function
def compute_scene_pacing_metrics(...):
    # This stays unchanged
```

**Implementation Steps**:
1. Comment out lines 4480-4517 in precompute_functions_full.py
2. Remove `run_claude_prompt` import if present
3. Test that scene_pacing still works
4. Delete commented code after verification

### Phase 2: Fix Terminology Consistency (Low Risk)
**Goal**: Standardize on "scenes" terminology

```python
# In compute_scene_pacing_metrics, change return keys:
return {
    'total_scenes': total_scenes,  # Was 'total_shots'
    'avg_scene_duration': avg_scene_duration,  # Was 'avg_shot_duration'
    'scenes_per_minute': scenes_per_minute,  # Was 'shots_per_minute'
    # Keep internal variable names for safety
}
```

**Implementation Steps**:
1. Update return dictionary keys only
2. Update professional wrapper to use new keys
3. Keep internal computation variables unchanged
4. Test output format remains valid

### Phase 3: Simplify Timestamp Handling (Medium Risk)
**Goal**: Reduce format conversions

```python
# Keep numeric timestamps longer
def _extract_timelines_from_analysis(analysis_dict):
    # Store numeric values directly
    sceneChangeTimeline = {}
    for change_time in scene_changes:
        # Simple integer key instead of "X-Ys" format
        key = f"scene_{int(change_time)}"
        sceneChangeTimeline[key] = {
            'time': change_time,  # Keep precise numeric value
            'type': 'scene_change'
        }
```

**Note**: This requires updating consumers of sceneChangeTimeline

## What NOT to Optimize

### Don't Touch:
1. **PySceneDetect integration** - Working well with adaptive thresholds
2. **Core metrics computation** - Complex but functional
3. **6-block professional format** - Required for compatibility
4. **Timeline entry creation** - Used by other systems

### Don't Remove:
1. **Dual timeline structures** - May have hidden dependencies
2. **Professional wrapper defaults** - Provides fallback values
3. **FPS manager imports** - May be used in future

## Expected Improvements

### Phase 1 Benefits:
- Remove ~40 lines of dead Claude API code
- Cleaner codebase
- Eliminate confusion about API usage

### Phase 2 Benefits:
- Consistent terminology
- Eliminate mapping layer
- Clearer code intent

### Phase 3 Benefits:
- Fewer string operations
- Preserve timestamp precision
- Reduced memory usage

## Testing Strategy

### Phase 1 Testing:
```bash
# Verify scene_pacing still works
python scripts/rumiai_runner.py https://tiktok.com/video

# Check output format unchanged
diff old_scene_pacing.json new_scene_pacing.json
```

### Phase 2 Testing:
```bash
# Verify terminology updates work
grep -r "total_shots" .  # Should only appear in comments
grep -r "total_scenes" .  # Should appear in outputs
```

## Conclusion

The scene_pacing flow's main issue is **Claude API optimization code for a system that doesn't use Claude**. This creates confusion and maintenance overhead without any benefit.

The safest optimization is:
1. **Remove Claude API code** (zero risk, immediate clarity)
2. **Fix terminology** (low risk, better consistency)
3. **Consider timestamp simplification** (medium risk, defer if uncertain)

Unlike visual_overlay which had processing redundancy, scene_pacing's issue is primarily **dead code and terminology confusion** rather than performance problems. The actual scene detection and metrics computation work well and should be preserved.