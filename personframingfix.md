# Person Framing Fix - Implementation Plan

## 🎯 Purpose: Infrastructure Fix for average_face_size

This fix is a **prerequisite infrastructure correction** needed before implementing `average_face_size`. We must first resolve the data flow conflicts and architectural inconsistencies to ensure a clean, reliable foundation for the feature.

## 🔍 Discovery Summary

### The Problem
1. **Missing Feature**: `average_face_size` metric not available in temporal windows
2. **Data Flow Confusion**: Face data has two conflicting paths in temporal_compute.py (BLOCKS average_face_size)
3. **Architecture Mismatch**: Mixed references to old (precompute) and new (temporal) systems

### Key Discoveries

#### 1. Timeline_builder is ACTIVE Infrastructure
```python
# In rumiai_runner.py:
self.timeline_builder = TimelineBuilder()
unified_analysis = self.timeline_builder.build_timeline(
    video_id,
    video_metadata.to_dict(),
    ml_results
)
```
- **NOT part of precompute** (which we're deleting)
- Part of the main pipeline that creates unified_analysis
- Converts ML results → standardized timeline entries
- Runs BEFORE temporal_compute.py

#### 2. Face Data Has Duplicate Paths (VERIFIED BUG!)

**Investigation Conducted:** Tested 5 videos to understand data sources:
- Timeline entries: Contains complete face data (105 faces in test video)
- ML data: Contains identical face data (105 faces)
- **Line 378 is redundant** - overwrites with same data

```python
# Path 1: Timeline entries (lines 302-310)
face_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'face':
        face_timeline.append({
            'timestamp': entry.get('start', 0),
            'bbox': entry.get('data', {}).get('bbox', {}),
            'confidence': entry.get('data', {}).get('confidence', 0)
        })
timelines['face_timeline'] = face_timeline  # Has 105 faces

# Path 2: ML data (line 378) - OVERWRITES Path 1!
mediapipe_data = extract_mediapipe_data(ml_data)
timelines['face_timeline'] = mediapipe_data.get('faces', [])  # Replaces with same 105 faces!
```

**Data Loss from Overwrite:**
- Timeline path produces: `{timestamp, bbox, confidence}`
- ML path produces: `{timestamp, bbox, confidence, count, frame_number}`
- We lose `count` and `frame_number` fields, but these are unused in the codebase

#### 3. Data Structure Verification
```python
# Timeline entries structure (created by timeline_builder):
{
    "entry_type": "face",
    "start": 0.0,
    "data": {
        "bbox": {"x": 0.31, "y": 0.22, "width": 0.33, "height": 0.18},
        "confidence": 0.96
    }
}

# ML data structure (raw from MediaPipe):
{
    "timestamp": 0.0,
    "bbox": {"x": 0.31, "y": 0.22, "width": 0.33, "height": 0.18},
    "confidence": 0.96,
    "frame_number": 0,
    "count": 1
}

# After extraction to face_timeline, both become:
{
    "timestamp": 0.0,
    "bbox": {"x": 0.31, "y": 0.22, "width": 0.33, "height": 0.18},
    "confidence": 0.96
}
```

## 🏗️ Two-Phase Implementation Strategy

### Why Two Phases?
The current face data path conflict makes it unsafe to add `average_face_size` directly:
- **Line 378 overwrites timeline data** → Face data source is unreliable
- **bbox access pattern inconsistent** → Can't guarantee data structure
- **No single source of truth** → Results would be unpredictable
- **Missing validation** → No detection of timeline_builder failures

Therefore, we must:
1. **Phase 1**: Fix infrastructure with fail-fast validation to ensure data integrity
2. **Phase 2**: Implement average_face_size feature on validated foundation

## 🎯 Implementation Plan

### Phase 1: Fix Face Data Path Conflict (Infrastructure)

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`

**Step 1: Implement fail-fast validation** (around line 378)
```python
# REPLACE the overwriting line with fail-fast validation:
ml_faces = mediapipe_data.get('faces', [])

# Validate that timeline_builder processed faces correctly
if ml_faces and not face_timeline:
    # CRITICAL: Timeline builder failed to process faces!
    raise ValueError(f"Data integrity error: Timeline builder missing {len(ml_faces)} faces found in ML data. "
                    f"This indicates a bug in timeline_builder that must be fixed.")
elif not ml_faces and not face_timeline:
    # Correct case - video genuinely has no faces
    logger.debug("No faces detected in video - both sources agree")
    face_timeline = []

# Remove the old overwriting line:
# timelines['face_timeline'] = mediapipe_data.get('faces', [])

# Use the validated timeline data
timelines['face_timeline'] = face_timeline
```

**Step 2: Fix bbox access pattern** (lines 302-310)
Since timeline_builder creates proper face entries, update extraction:
```python
# Current (partially wrong):
face_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'face':
        face_timeline.append({
            'timestamp': entry.get('start', 0),
            'bbox': entry.get('data', {}).get('bbox', {}),
            'confidence': entry.get('data', {}).get('confidence', 0)
        })

# Fixed (consistent with timeline structure):
face_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'face':
        face_data = entry.get('data', {})
        if face_data.get('bbox'):  # Only add if bbox exists
            face_timeline.append({
                'timestamp': entry.get('start', 0),
                'bbox': face_data.get('bbox', {}),
                'confidence': face_data.get('confidence', 0),
                'count': face_data.get('count', 1)  # For multi-face support
            })
```

### Phase 2: Add average_face_size Metric (Feature)

**Location**: In `process_segment()` function (around lines 1320-1360)

**Step 1: Collect face areas during existing loop**
```python
# After line 1324, add:
face_areas = []  # Collect areas for average calculation

# Inside the loop (after line 1334), add:
if bbox:
    face_area = bbox.get('width', 0) * bbox.get('height', 0) * 100
    face_areas.append(face_area)  # Store for averaging

    # Existing categorization continues...
    if face_area > 25:
        framing_counts['close'] += 1
    # etc...
```

**Step 2: Calculate average after loop**
```python
# After line 1349 (after the loop), add:
# Calculate average face size
average_face_size = sum(face_areas) / len(face_areas) if face_areas else 0.0
```

**Step 3: Add to return dictionary**
```python
# Around line 1427, modify return to include:
return {
    # ... existing metrics ...
    **framing_dist,
    'average_face_size': round(average_face_size, 4),  # Add this line
    # ... rest of metrics ...
}
```

## 📊 Expected Output

### Before Fix:
```json
{
  "temporal_windows": {
    "hook": {
      "close_ratio": 0.0,
      "medium_ratio": 1.0,
      "wide_ratio": 0.0,
      "none_ratio": 0.0
      // No average_face_size
    }
  }
}
```

### After Fix:
```json
{
  "temporal_windows": {
    "hook": {
      "close_ratio": 0.0,
      "medium_ratio": 1.0,
      "wide_ratio": 0.0,
      "none_ratio": 0.0,
      "average_face_size": 0.1234  // NEW: 12.34% of frame
    }
  }
}
```

## 🔄 Infrastructure Validation

### Why This Order Matters
Without Phase 1 (infrastructure fix):
```python
# Current broken state:
timeline_faces = [entry1, entry2, ...]  # Created from timeline entries
timelines['face_timeline'] = timeline_faces
# ... later ...
timelines['face_timeline'] = ml_faces  # OVERWRITES! Now using different data

# Result: average_face_size would calculate from ML data
# But framing_ratios might use timeline data
# Inconsistent metrics!
```

With Phase 1 fixed (fail-fast approach):
```python
# Fixed state with validation:
timeline_faces = [entry1, entry2, ...]  # Created from timeline entries
ml_faces = mediapipe_data.get('faces', [])

# Validation ensures data integrity
if ml_faces and not timeline_faces:
    raise ValueError(f"Timeline builder bug: missing {len(ml_faces)} faces")

timelines['face_timeline'] = timeline_faces  # Single source of truth

# Result:
# - Both average_face_size and framing_ratios use same data
# - Any timeline_builder failures are caught immediately
# - Consistent, reliable metrics!
```

## 🧪 Testing Plan

### Test 1: Verify Data Sources and Fail-Fast Behavior
```python
#!/usr/bin/env python3
import json
from pathlib import Path
from rumiai_v2.processors.temporal_compute import extract_timelines_for_temporal

# Load test video
test_file = Path('unified_analysis/7430952519439846698.json')
with open(test_file) as f:
    data = json.load(f)

# Test BEFORE fix to understand current state
timeline_entries = data.get('timeline', {}).get('entries', [])
ml_data = data.get('ml_data', {})

# Count faces from both sources
timeline_faces = [e for e in timeline_entries if e.get('entry_type') == 'face']
ml_faces = ml_data.get('mediapipe', {}).get('faces', [])

print(f"Timeline faces: {len(timeline_faces)}")
print(f"ML faces: {len(ml_faces)}")

if ml_faces and not timeline_faces:
    print("ERROR: This would trigger fail-fast after fix!")
    print(f"Timeline builder missing {len(ml_faces)} faces")
elif not ml_faces and not timeline_faces:
    print("OK: Video has no faces (both sources agree)")
else:
    print("OK: Both sources have face data")

# Verify data structure consistency
if timeline_faces and ml_faces:
    print(f"\nTimeline face structure: {list(timeline_faces[0].keys())}")
    print(f"ML face structure: {list(ml_faces[0].keys())}")
```

### Test 2: Verify average_face_size Calculation
```python
#!/usr/bin/env python3
from rumiai_v2.processors.temporal_compute import compute_temporal_windows
import json
from pathlib import Path

# Process video
test_file = Path('unified_analysis/7430952519439846698.json')
with open(test_file) as f:
    data = json.load(f)

result = compute_temporal_windows(data)

# Check all windows have average_face_size
for window_type in ['hook', 'closing']:
    window = result['temporal_windows'].get(window_type)
    if window:
        print(f"{window_type}:")
        print(f"  average_face_size: {window.get('average_face_size', 'MISSING')}")
        print(f"  close_ratio: {window.get('close_ratio')}")

for segment in result['temporal_windows'].get('middle_segments', []):
    print(f"Middle segment {segment.get('start')}-{segment.get('end')}:")
    print(f"  average_face_size: {segment.get('average_face_size', 'MISSING')}")
```

## 🎯 Success Criteria

1. ✅ Face timeline uses consistent data source (timeline entries)
2. ✅ No data overwriting between paths
3. ✅ `average_face_size` appears in all temporal windows
4. ✅ Values are reasonable (0.0-1.0 range, typically 0.05-0.40)
5. ✅ Correlates with framing ratios (high close_ratio = high average_face_size)

## 📈 ML Value

### What average_face_size Provides:
- **Continuous magnitude** vs categorical ratios
- **Exact prominence**: Distinguishes "barely close" (26%) from "extreme close-up" (60%)
- **Progression tracking**: Smooth zoom in/out patterns
- **Intensity measure**: How intimate/distant the framing is

### Example Patterns ML Can Learn:
```python
# Beauty tutorial pattern
Hook: average_face_size=0.15 → Middle: 0.35 → Close: 0.40
"Gradual zoom to product demonstration"

# Storytime pattern
Hook: average_face_size=0.25 → Middle: 0.22 → Close: 0.23
"Consistent medium framing for narrative"

# Product review pattern
Hook: average_face_size=0.30 → Middle: 0.08 → Close: 0.35
"Face intro → product focus → face outro"
```

## ⚠️ Risks & Mitigations

### Risk 1: Breaking Existing Pipeline
- **Mitigation**: Fail-fast validation will catch issues immediately rather than silently failing
- **Rollback**: If too many videos fail validation, temporarily switch to warning instead of error:
  ```python
  # Temporary degraded mode if needed:
  if ml_faces and not face_timeline:
      logger.warning(f"Timeline missing {len(ml_faces)} faces - using ML data")
      face_timeline = ml_faces  # Temporary fallback
  ```

### Risk 2: Timeline Builder Bugs
- **Detection**: Fail-fast will identify any videos where timeline_builder fails to process faces
- **Mitigation**: Fix timeline_builder bugs as they're discovered
- **Validation**: Error messages clearly indicate the problem source

### Risk 3: Performance Impact
- **Mitigation**: Validation adds negligible overhead (simple list length check)
- **Benefit**: Early detection prevents cascading failures

## 📅 Implementation Steps

### Phase 1: Infrastructure (Must Complete First)
1. **Backup current temporal_compute.py** (2 min)
2. **Implement fail-fast validation** - Replace line 378 with validation logic (5 min)
3. **Update bbox extraction logic** - Ensure consistent access (5 min)
4. **Test fail-fast behavior** - Verify errors trigger for missing timeline data (10 min)
5. **Validate single source of truth** - Confirm only timeline data is used (5 min)

### Phase 2: Feature Implementation (Only After Phase 1)
1. **Add face_areas collection** in existing loop (5 min)
2. **Calculate average_face_size** after loop (3 min)
3. **Add to return dictionary** (2 min)
4. **Run integration tests** (10 min)
5. **Update ImprovementsMLMVP.md** (2 min)

**Phase 1 time**: ~27 minutes (Infrastructure)
**Phase 2 time**: ~22 minutes (Feature)
**Total time**: ~49 minutes

## 🚀 Decision

**IMPLEMENT IN TWO PHASES** - This approach:
- **Phase 1**: Resolves architectural inconsistency (infrastructure foundation)
- **Phase 2**: Adds valuable ML feature (average_face_size)
- Ensures data consistency before feature addition
- Minimal risk with clear rollback points
- Aligns with timeline_builder architecture

### Critical Success Factor
**Phase 1 MUST be completed and validated before Phase 2**. Adding average_face_size without fixing the infrastructure would result in:
- Inconsistent metrics (framing_ratios vs average_face_size from different sources)
- Unpredictable behavior (which data source wins depends on code execution order)
- Difficult debugging (issues could appear intermittently)

### Expected Timeline
- Day 1: Complete Phase 1 (infrastructure), validate
- Day 2: Implement Phase 2 (feature), full testing
- Result: Clean, reliable average_face_size implementation