# Multi-person Metrics MVP - person_count Implementation

## Feature: person_count

### Current Problem
- `object_count` includes ALL objects (person, laptop, cup, etc.)
- Cannot distinguish human presence from object presence
- No way to identify videos with people vs. product-only content

### MVP Goal
Add a single metric: **`person_count`** - total person detections in each temporal window

### Data Source (Verified)
- **YOLO object detection** already provides person detections
- Available in `timelines['object_timeline']` 
- Each detection has `className: "person"`
- Already integrated into temporal windows pipeline

## Implementation Roadmap

### Step 1: Add person_count to process_segment()

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`

**Find exact location**:
```bash
grep -n "object_count = len(segment_objects)" rumiai_v2/processors/temporal_compute.py
# Output: 566:    object_count = len(segment_objects)
```

**Current Code** (line 566):
```python
segment_objects = [o for o in timelines.get('object_timeline', [])
                  if start <= o.get('timestamp', 0) < end]
object_count = len(segment_objects)
```

**Add After** (line 567):
```python
# MVP: Count person detections specifically
# Note: No new imports required - file already has all necessary imports
# Note: YOLO always outputs lowercase 'person' (verified across 3000+ instances)
segment_persons = [o for o in segment_objects 
                  if o.get('className') == 'person']
person_count = len(segment_persons)
```

### Step 2: Add to Return Dictionary

**Find exact location**:
```bash
grep -n "'object_count':" rumiai_v2/processors/temporal_compute.py
# Output: 743:        'object_count': object_count,
```

**Current Return Structure** (lines 741-747):
```python
        'text_count': text_count,
        'sticker_count': sticker_count,
        'object_count': object_count,
        'gesture_count': gesture_count,
        'expression_count': expression_count,
        'scene_count': scene_count,
        'element_count': total_elements,
```

**Modified Return** (add after `'object_count': object_count,`):
```python
        'text_count': text_count,
        'sticker_count': sticker_count,
        'object_count': object_count,
        'person_count': person_count,  # NEW: Person-specific count
        'gesture_count': gesture_count,
        'expression_count': expression_count,
        'scene_count': scene_count,
        'element_count': total_elements,
```

### Step 3: Expected Output

```json
{
  "temporal_windows": {
    "hook": {
      "object_count": 30,    // All YOLO objects
      "person_count": 25,    // Just persons (NEW)
      ...
    },
    "middle_segments": [...],
    "closing": {...}
  }
}
```

### Step 4: Test Implementation

```bash
# Test with single-person video (baseline)
python3 test_temporal_compute_v2.py 7430952519439846698

# Test with MULTI-PERSON video (validation)
# Video: https://www.tiktok.com/@katekate1329/video/7529056602041699592
# Note: First run full pipeline to generate ML outputs:
# python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@katekate1329/video/7529056602041699592'
# Then test:
python3 test_temporal_compute_v2.py 7529056602041699592

# Verify person_count appears and is <= object_count
grep "person_count" test_outputs/*_temporal_windows_test.json
```

### Step 5: Validate with ML

- Check if `person_count` provides signal for engagement
- Compare correlation with `object_count`
- Determine if separation improves model performance

## Why This MVP Makes Sense

1. **Minimal Change**: ~3 lines of code
2. **No Breaking Changes**: Adds field without removing anything
3. **Clear Value**: Distinguishes human vs. non-human content
4. **Already Supported**: Uses existing YOLO data
5. **Easy to Test**: Simple count comparison

## Future Iterations (NOT MVP)

Once person_count proves valuable, consider:
- Phase 2: `has_person` boolean (person_count > 0)
- Phase 3: `max_persons_per_frame` (requires timestamp grouping)  
- Phase 4: `solo_ratio`, `multi_person_ratio` (requires unique person tracking)
- Phase 5: Animal detection (`animal_count`)

But **ONLY** after proving person_count adds ML value.

## Implementation Checklist

- [ ] Add person filtering logic (3 lines including comments)
- [ ] Add person_count to return dict (1 line)
- [ ] Run test with single-person video
- [ ] Run test with multi-person video
- [ ] Check person_count <= object_count
- [ ] Update ImprovementsMLMVP.md status

## Quick Verification

After running the test script, verify person_count was added:

```bash
# Check if person_count exists in output
grep -c "person_count" test_outputs/*_temporal_windows_test.json

# View person_count vs object_count for all windows
grep -E "(person_count|object_count)" test_outputs/*_temporal_windows_test.json | head -20
```

## Success Criteria

✅ MVP is successful if:
1. `person_count` field appears in all temporal windows
2. `person_count <= object_count` always
3. Test videos with people show `person_count > 0`
4. No performance degradation

## Risk Assessment

- **Risk**: None - just filtering existing data
- **Rollback**: Remove 3 lines if needed
- **Performance**: Measured at <0.002ms for typical windows (50-100 objects)
  - 50 objects: 0.0009ms
  - 100 objects: 0.0018ms  
  - 500 objects: 0.0104ms
- **Compatibility**: Fully backward compatible

---

## Future Features (After MVP Validation)

### Phase 2: Basic Person Presence
Once `person_count` proves valuable, add:

**`has_person`** - Boolean indicator
```python
has_person = person_count > 0
```

### Phase 3: Multi-Person Tracking
Requires verifying trackID behavior and timestamp grouping:

```python
# Track unique persons per timestamp
persons_per_timestamp = {}
for person in segment_persons:
    timestamp = person.get('timestamp', 0)
    track_id = person.get('trackId', 'unknown')
    
    # Group by timestamp (needs verification of optimal grouping)
    timestamp_key = round(timestamp, 1)  # Or use frame-level grouping
    
    if timestamp_key not in persons_per_timestamp:
        persons_per_timestamp[timestamp_key] = set()
    persons_per_timestamp[timestamp_key].add(track_id)

# Derive metrics
person_counts = [len(tracks) for tracks in persons_per_timestamp.values()]
avg_person_count = float(np.mean(person_counts))
max_person_count = int(max(person_counts))
min_person_count = int(min(person_counts))
```

### Phase 4: Collaboration Patterns
After confirming multi-person tracking works:

**Distribution Metrics**:
- `solo_ratio` - % of timestamps with 1 person
- `duo_ratio` - % of timestamps with 2 people  
- `group_ratio` - % of timestamps with 3+ people
- `has_collaboration` - Boolean (max_person_count > 1)

**Consistency Metric**:
```python
# Needs proper normalization formula
if max_person_count > min_person_count:
    person_std = float(np.std(person_counts))
    # Better formula needed - current one can go negative
    collaboration_consistency = 1.0 - (person_std / (max_person_count - min_person_count))
else:
    collaboration_consistency = 1.0
```

### Phase 5: Object Type Analysis
Leverage YOLO's 56+ object classes:

**Animal Detection**:
```python
animal_classes = {'dog', 'cat', 'bird'}
segment_animals = [o for o in segment_objects if o.get('className') in animal_classes]
has_animals = len(segment_animals) > 0
animal_count = len(set(a.get('trackId') for a in segment_animals))
```

**Content Type Indicators**:
```python
# Food content
food_classes = {'pizza', 'cake', 'donut', 'banana', 'apple', 'sandwich'}
has_food = any(o.get('className') in food_classes for o in segment_objects)

# Tech content  
tech_classes = {'laptop', 'cell phone', 'keyboard', 'mouse', 'tv'}
tech_count = len([o for o in segment_objects if o.get('className') in tech_classes])

# Vehicle content
vehicle_classes = {'car', 'motorcycle', 'boat', 'train'}
has_vehicles = any(o.get('className') in vehicle_classes for o in segment_objects)
```

### Phase 6: Advanced Metrics
Only after all above phases prove valuable:

**Temporal Patterns**:
- Person appearance/disappearance timestamps
- Collaboration transitions (solo→group, group→solo)
- Peak crowding moments

**Interaction Indicators**:
- Distance between persons (requires bbox analysis)
- Face-to-face detection (combine with MediaPipe)
- Group formation patterns

### Complete Feature List (All Phases)

| Phase | Metric | Type | Description | Complexity |
|-------|--------|------|-------------|-----------|
| **1 (MVP)** | `person_count` | int | Total person detections | Trivial |
| 2 | `has_person` | bool | Any person detected | Trivial |
| 3 | `avg_person_count` | float | Average unique persons per timestamp | Medium |
| 3 | `max_person_count` | int | Maximum persons in any frame | Medium |
| 3 | `min_person_count` | int | Minimum persons detected | Medium |
| 4 | `solo_ratio` | float | % frames with 1 person | Medium |
| 4 | `duo_ratio` | float | % frames with 2 people | Medium |
| 4 | `group_ratio` | float | % frames with 3+ people | Medium |
| 4 | `has_collaboration` | bool | Multiple people detected | Easy |
| 4 | `collaboration_consistency` | float | Person count stability | Hard |
| 5 | `has_animals` | bool | Animals detected | Easy |
| 5 | `animal_count` | int | Unique animals | Medium |
| 5 | `has_food` | bool | Food detected | Easy |
| 5 | `tech_count` | int | Tech objects count | Easy |
| 5 | `has_vehicles` | bool | Vehicles detected | Easy |
| 6 | `collaboration_transitions` | int | Solo↔Group changes | Hard |
| 6 | `peak_crowd_timestamp` | float | When most people appear | Medium |

### Implementation Prerequisites

Before implementing each phase:

**Phase 3 Prerequisites**:
- Verify trackID uniqueness across frames
- Determine optimal timestamp grouping strategy
- Test with known multi-person videos

**Phase 4 Prerequisites**:
- Confirm Phase 3 metrics are accurate
- Validate ratios sum to 1.0
- Define "collaboration" semantically

**Phase 5 Prerequisites**:
- Test YOLO animal detection accuracy
- Verify object classes in production videos
- Consider false positive rates

**Phase 6 Prerequisites**:
- All previous phases stable
- Clear ML value demonstrated
- Performance impact acceptable

### Why Phased Approach

1. **Risk Mitigation**: Each phase proves value before next
2. **Complexity Management**: Start simple, add only what works
3. **Performance**: Monitor impact at each stage
4. **Rollback**: Easy to remove unsuccessful features
5. **Learning**: Each phase informs the next

The key principle: **Don't implement Phase N+1 until Phase N proves valuable**