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
- **Lines 375-377 are redundant** - extract_mediapipe_data overwrites with same data

```python
# Correct Path: MediaPipe → timeline_builder → timeline entries
face_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'face':
        face_timeline.append({
            'timestamp': entry.get('start', 0),
            'bbox': entry.get('data', {}).get('bbox', {}),
            'confidence': entry.get('data', {}).get('confidence', 0)  # Currently extracted but unused
        })
timelines['face_timeline'] = face_timeline  # Has 105 faces from timeline

# Redundant Path (lines 375-377) - OVERWRITES with same data!
mediapipe_data = extract_mediapipe_data(ml_data)  # Line 375
timelines['pose_timeline'] = mediapipe_data.get('poses', [])  # Line 376 - UNUSED
timelines['face_timeline'] = mediapipe_data.get('faces', [])  # Line 377 - Overwrites!
```

**Data Structure Analysis:**
- Timeline currently extracts: `{timestamp, bbox, confidence}`
- ML path has: `{timestamp, bbox, confidence, count, frame_number}`
- **Only `timestamp` and `bbox` are actually used** in temporal_compute.py
- Decision: Simplify to extract only what we need

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

# After simplified extraction (only what we use):
{
    "timestamp": 0.0,
    "bbox": {"x": 0.31, "y": 0.22, "width": 0.33, "height": 0.18}
    # Removed: confidence, count, frame_number (all unused)
}
```

## 🏗️ Two-Phase Implementation Strategy

### Why Two Phases?
The current face data path conflict makes it unsafe to add `average_face_size` directly:
- **Lines 375-377 create duplicate data paths** → Face data source is unreliable
- **Unused fields being extracted** → Wasting resources on confidence, count, frame_number
- **No single source of truth** → Results would be unpredictable
- **Missing validation** → No detection of timeline_builder failures

Therefore, we must:
1. **Phase 1**: Fix infrastructure with fail-fast validation to ensure data integrity
2. **Phase 2**: Implement average_face_size feature on validated foundation

## 🎯 Implementation Plan

### Phase 1: Fix Face Data Path Conflict (Infrastructure)

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`

**Step 1: Remove MediaPipe extraction and add validation** (lines 375-377 as of Jan 2025)
```python
# DELETE these lines entirely:
# mediapipe_data = extract_mediapipe_data(ml_data)
# timelines['pose_timeline'] = mediapipe_data.get('poses', [])  # Unused
# timelines['face_timeline'] = mediapipe_data.get('faces', [])  # Overwrites timeline

# ADD validation to ensure timeline_builder processed faces:
# (Note: face_timeline was already extracted from timeline entries above)

# Get ML face count for validation only
ml_mediapipe = ml_data.get('mediapipe', {})
ml_faces = ml_mediapipe.get('faces', [])

# Validate that timeline_builder processed faces correctly
if ml_faces and not face_timeline:
    # CRITICAL: Timeline builder failed to process MediaPipe faces!
    raise ValueError(f"Data integrity error: Timeline builder missing {len(ml_faces)} faces from MediaPipe. "
                    f"This indicates a bug in timeline_builder that must be fixed.")
elif not ml_faces and not face_timeline:
    # Correct case - video genuinely has no faces
    logger.debug("No faces detected in video - both sources agree")

# face_timeline is already set from timeline entries extraction above
# Single source of truth: MediaPipe → timeline_builder → timeline entries → face_timeline
```

**Step 2: Simplify extraction to only used fields** (starting at line 302 as of Jan 2025)
```python
# Current (extracting unused fields):
face_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'face':
        face_timeline.append({
            'timestamp': entry.get('start', 0),
            'bbox': entry.get('data', {}).get('bbox', {}),
            'confidence': entry.get('data', {}).get('confidence', 0)  # UNUSED
        })

# Simplified (only extract what we actually use):
face_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'face':
        face_data = entry.get('data', {})
        bbox = face_data.get('bbox')
        if bbox:  # Only add if bbox exists
            face_timeline.append({
                'timestamp': entry.get('start', 0),
                'bbox': bbox
                # Excluded: confidence (unused), count (unused), frame_number (unused)
            })
```

### Phase 2: Add average_face_size Metric (Feature)

**Location**: In `process_segment()` function (around lines 1320-1360 as of Jan 2025)

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

### Test 1: Diagnose Current Overwrite Bug (BEFORE fix)
```python
#!/usr/bin/env python3
"""Test to demonstrate the current overwrite bug before fixing."""
import json
from pathlib import Path
import sys
sys.path.append('/home/jorge/rumiaifinal')
from rumiai_v2.processors.temporal_compute import extract_timelines_for_temporal

# Load test video
test_file = Path('unified_analysis/7430952519439846698.json')
with open(test_file) as f:
    data = json.load(f)

# Temporarily modify extract_timelines_for_temporal to log both sources
print("=== BEFORE FIX - Demonstrating the overwrite bug ===\n")

# Count from both sources
timeline_entries = data.get('timeline', {}).get('entries', [])
timeline_faces = [e for e in timeline_entries if e.get('entry_type') == 'face']
ml_faces = data.get('ml_data', {}).get('mediapipe', {}).get('faces', [])

print(f"Timeline path would extract: {len(timeline_faces)} faces")
print(f"ML path would extract: {len(ml_faces)} faces")

# Show what happens at line 310 vs line 377
timelines = extract_timelines_for_temporal(data)
face_timeline_after = timelines.get('face_timeline', [])
print(f"\nFinal face_timeline has: {len(face_timeline_after)} faces")
print(f"Which source won? Check if structure has 'frame_number' field:")
if face_timeline_after:
    has_frame_number = 'frame_number' in face_timeline_after[0]
    print(f"  Has frame_number field: {has_frame_number}")
    print(f"  → Data is from: {'ML path (BUG!)' if has_frame_number else 'Timeline path'}")
```

### Test 2: Verify Fix Works (AFTER fix)
```python
#!/usr/bin/env python3
"""Test to verify the fix eliminates the overwrite."""
import json
from pathlib import Path
import sys
sys.path.append('/home/jorge/rumiaifinal')

# Mock the fixed version
def test_fixed_extraction(data):
    """Simulates the fixed extraction logic."""
    timeline_entries = data.get('timeline', {}).get('entries', [])
    ml_data = data.get('ml_data', {})

    # Extract faces from timeline only
    face_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'face':
            face_data = entry.get('data', {})
            bbox = face_data.get('bbox')
            if bbox:
                face_timeline.append({
                    'timestamp': entry.get('start', 0),
                    'bbox': bbox  # Simplified - only what we use
                })

    # Validation (fail-fast)
    ml_faces = ml_data.get('mediapipe', {}).get('faces', [])
    if ml_faces and not face_timeline:
        raise ValueError(f"Timeline builder bug: missing {len(ml_faces)} faces")

    return face_timeline

# Test with multiple videos
test_videos = [
    'unified_analysis/7430952519439846698.json',  # Has faces
    # Add more test videos as needed
]

print("=== AFTER FIX - Verifying single source of truth ===\n")

for video_path in test_videos:
    print(f"Testing: {video_path}")
    with open(video_path) as f:
        data = json.load(f)

    try:
        face_timeline = test_fixed_extraction(data)
        print(f"  ✓ Extracted {len(face_timeline)} faces from timeline")
        if face_timeline:
            print(f"  ✓ Structure: {list(face_timeline[0].keys())}")
    except ValueError as e:
        print(f"  ✗ FAIL-FAST: {e}")
```

### Test 3: Edge Cases and Validation
```python
#!/usr/bin/env python3
"""Test edge cases to ensure robustness."""

def test_edge_cases():
    """Test various edge cases."""

    # Case 1: No faces at all
    print("Test 1: Video with no faces")
    data_no_faces = {
        'timeline': {'entries': []},
        'ml_data': {'mediapipe': {'faces': []}}
    }
    result = test_fixed_extraction(data_no_faces)
    assert len(result) == 0, "Should handle no faces gracefully"
    print("  ✓ Passed\n")

    # Case 2: Timeline builder failure (faces in ML but not timeline)
    print("Test 2: Timeline builder failure")
    data_timeline_bug = {
        'timeline': {'entries': []},  # No face entries!
        'ml_data': {'mediapipe': {'faces': [
            {'timestamp': 0, 'bbox': {'x': 0.3, 'y': 0.3, 'width': 0.2, 'height': 0.3}}
        ]}}
    }
    try:
        result = test_fixed_extraction(data_timeline_bug)
        print("  ✗ Should have raised an error!")
    except ValueError as e:
        print(f"  ✓ Correctly caught: {e}\n")

    # Case 3: Faces without bbox (malformed data)
    print("Test 3: Malformed face data (no bbox)")
    data_malformed = {
        'timeline': {'entries': [
            {'entry_type': 'face', 'start': 0, 'data': {}}  # No bbox!
        ]},
        'ml_data': {'mediapipe': {'faces': []}}
    }
    result = test_fixed_extraction(data_malformed)
    assert len(result) == 0, "Should skip faces without bbox"
    print("  ✓ Passed - skipped malformed entries\n")

if __name__ == "__main__":
    test_edge_cases()
```

### Test 4: Verify average_face_size Feature (Phase 2)
```python
#!/usr/bin/env python3
"""Test the average_face_size feature after Phase 2 implementation."""
from rumiai_v2.processors.temporal_compute import compute_temporal_windows
import json
from pathlib import Path

# Process video
test_file = Path('unified_analysis/7430952519439846698.json')
with open(test_file) as f:
    data = json.load(f)

result = compute_temporal_windows(data)

print("=== PHASE 2 - Testing average_face_size feature ===\n")

# Check all windows have average_face_size
for window_type in ['hook', 'closing']:
    window = result['temporal_windows'].get(window_type)
    if window:
        avg_size = window.get('average_face_size', 'MISSING')
        close_ratio = window.get('close_ratio', 0)
        print(f"{window_type}:")
        print(f"  average_face_size: {avg_size}")
        print(f"  close_ratio: {close_ratio}")

        # Validate correlation
        if avg_size != 'MISSING':
            if avg_size > 0.25 and close_ratio < 0.5:
                print("  ⚠️ Warning: Large face size but low close_ratio - check calculation")
            elif avg_size < 0.08 and close_ratio > 0.5:
                print("  ⚠️ Warning: Small face size but high close_ratio - check calculation")
            else:
                print("  ✓ Face size and framing ratios correlate")

for segment in result['temporal_windows'].get('middle_segments', []):
    avg_size = segment.get('average_face_size', 'MISSING')
    print(f"\nMiddle segment {segment.get('start')}-{segment.get('end')}:")
    print(f"  average_face_size: {avg_size}")
    if avg_size == 'MISSING':
        print("  ✗ Feature not implemented yet")
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

### Risk Assessment Conducted (Jan 2025)
**Investigation Results:**
- ✅ `face_timeline` only used within temporal_compute.py
- ✅ No consumers depend on `confidence`, `count`, or `frame_number` fields
- ✅ `pose_timeline` is created but never used anywhere
- ✅ Production test (`test_temporal_compute_v2.py`) has no dependencies on removed fields
- ✅ Output consumers (rumiai_runner.py) just save JSON without structure requirements

**Risk Level: VERY LOW** - No production dependencies on removed code

### Risk 1: Timeline Builder Failures
- **Detection**: Fail-fast validation will immediately catch if timeline_builder misses faces
- **Mitigation**: Clear error messages identify the exact problem
- **Rollback**: If needed, temporary degraded mode:
  ```python
  # Emergency fallback if too many videos fail:
  if ml_faces and not face_timeline:
      logger.warning(f"Timeline missing {len(ml_faces)} faces - using ML data")
      face_timeline = ml_faces  # Temporary fallback until timeline_builder fixed
  ```

### Risk 2: Test Suite Impact
- **Impact**: None - test_temporal_compute_v2.py doesn't use removed fields
- **Mitigation**: Tests will actually help validate the fix works correctly

### Risk 3: Performance Impact
- **Impact**: Positive - removing redundant extraction and unused fields
- **Validation**: Adds negligible overhead (simple list length check)

## 📅 Implementation Steps

### Phase 1: Infrastructure (Must Complete First)
1. **Backup current temporal_compute.py** (2 min)
2. **Remove MediaPipe extraction** - Delete lines 375-377 entirely (2 min)
3. **Add fail-fast validation** - Validate timeline_builder processed faces (5 min)
4. **Simplify face extraction** - Only extract timestamp and bbox fields (5 min)
5. **Test fail-fast behavior** - Verify errors trigger for missing timeline data (10 min)
6. **Validate single source** - Confirm only timeline path is used (3 min)

### Phase 2: Feature Implementation (Only After Phase 1)
1. **Add face_areas collection** in existing loop (5 min)
2. **Calculate average_face_size** after loop (3 min)
3. **Add to return dictionary** (2 min)
4. **Run integration tests** (10 min)
5. **Update ImprovementsMLMVP.md** (2 min)

**Phase 1 time**: ~27 minutes (Infrastructure with 6 steps)
**Phase 2 time**: ~22 minutes (Feature implementation)
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