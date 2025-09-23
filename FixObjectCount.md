# Fix Object Count: Track Unique Object Instances

## Problem Statement
Currently, `object_count` in temporal windows counts every single detection across all frames. If YOLO detects the same person in 10 frames, it counts as 10 objects. This is misleading - we want to count unique object instances.

### Current Behavior
```python
# In temporal_compute.py
object_count = len(segment_objects)  # Counts ALL detections
```

**Example**: Hook window (0-3s) shows `object_count: 17`
- 10 person detections (same person)
- 6 bottle detections (same bottle)
- 1 cup detection (same cup)

### Desired Behavior
`object_count: 3` (1 person + 1 bottle + 1 cup)

## Solution: Object Instance Tracking with IoU

### Core Algorithm
Track unique object instances by:
1. Objects in consecutive frames with IoU > threshold are the same instance
2. Objects that disappear and reappear are considered the same if within time window
3. Different objects of same class at different locations are unique instances

### Implementation Plan

#### Step 1: Create Object Tracker Class
**File**: `rumiai_v2/processors/object_tracker.py`

```python
class ObjectInstanceTracker:
    def __init__(self, iou_threshold=0.3, max_frames_missing=5):
        """
        Args:
            iou_threshold: Min IoU to consider same object (0.3 for movement tolerance)
            max_frames_missing: Frames object can be missing and still be same instance
        """
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        self.next_instance_id = 0
        self.active_instances = {}  # instance_id -> last_detection
        self.all_instances = {}     # instance_id -> all_detections

    def track_objects(self, detections_by_frame):
        """
        Process all detections and assign unique instance IDs

        Args:
            detections_by_frame: List of frame detections from YOLO

        Returns:
            Dict mapping instance_id -> {class, detections, time_range}
        """
        for frame_detections in detections_by_frame:
            self._process_frame(frame_detections)
        return self.all_instances

    def _process_frame(self, frame_detections):
        """Match detections to existing instances or create new ones"""
        matched_instances = set()

        for detection in frame_detections:
            best_match = self._find_best_match(detection)

            if best_match:
                # Update existing instance
                instance_id = best_match
                self.active_instances[instance_id] = detection
                self.all_instances[instance_id]['detections'].append(detection)
                matched_instances.add(instance_id)
            else:
                # Create new instance
                instance_id = self.next_instance_id
                self.next_instance_id += 1
                self.active_instances[instance_id] = detection
                self.all_instances[instance_id] = {
                    'class': detection['class'],
                    'detections': [detection],
                    'first_seen': detection['timestamp'],
                    'last_seen': detection['timestamp']
                }

        # Mark unmatched instances as inactive
        self._update_inactive_instances(matched_instances)

    def _find_best_match(self, detection):
        """Find best matching instance for detection"""
        best_iou = 0
        best_instance = None

        for instance_id, last_detection in self.active_instances.items():
            # Only match same class
            if last_detection['class'] != detection['class']:
                continue

            # Check temporal proximity
            time_diff = detection['timestamp'] - last_detection['timestamp']
            if time_diff > self.max_frames_missing * 0.33:  # Assuming ~3fps
                continue

            # Calculate IoU
            iou = self._calculate_iou(last_detection['bbox'], detection['bbox'])
            if iou > self.iou_threshold and iou > best_iou:
                best_iou = iou
                best_instance = instance_id

        return best_instance

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bboxes"""
        # bbox format: [x, y, width, height] or [x1, y1, x2, y2]
        # Convert to [x1, y1, x2, y2] if needed
        x1_min, y1_min, x1_max, y1_max = self._normalize_bbox(bbox1)
        x2_min, y2_min, x2_max, y2_max = self._normalize_bbox(bbox2)

        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
```

#### Step 2: Update Temporal Compute
**File**: `rumiai_v2/processors/temporal_compute.py`

```python
# Add import
from rumiai_v2.processors.object_tracker import ObjectInstanceTracker

# Replace object counting logic
def _compute_window_features(segment_objects, segment_gestures, ...):
    # OLD:
    # object_count = len(segment_objects)

    # NEW: Count unique object instances
    tracker = ObjectInstanceTracker(iou_threshold=0.3)

    # Group objects by frame
    objects_by_frame = {}
    for obj in segment_objects:
        frame_num = obj.get('frame_number', int(obj['timestamp'] * 3))  # Assuming 3fps
        if frame_num not in objects_by_frame:
            objects_by_frame[frame_num] = []
        objects_by_frame[frame_num].append({
            'class': obj.get('class', obj.get('className')),
            'bbox': obj['bbox'],
            'timestamp': obj['timestamp'],
            'confidence': obj.get('confidence', 1.0)
        })

    # Track instances
    unique_instances = tracker.track_objects(
        [objects_by_frame.get(i, []) for i in sorted(objects_by_frame.keys())]
    )

    object_count = len(unique_instances)

    # Also track unique persons separately (existing logic)
    person_instances = {
        iid: inst for iid, inst in unique_instances.items()
        if inst['class'] == 'person'
    }
    person_count = len(person_instances)
```

#### Step 3: Add Unit Tests
**File**: `tests/test_object_tracker.py`

```python
def test_same_object_tracked():
    """Test that same object across frames counts as 1"""
    tracker = ObjectInstanceTracker(iou_threshold=0.3)

    # Person detected in 3 consecutive frames with slight movement
    detections = [
        [{'class': 'person', 'bbox': [100, 100, 200, 300], 'timestamp': 0.0}],
        [{'class': 'person', 'bbox': [105, 100, 205, 300], 'timestamp': 0.33}],
        [{'class': 'person', 'bbox': [110, 100, 210, 300], 'timestamp': 0.66}],
    ]

    instances = tracker.track_objects(detections)
    assert len(instances) == 1, "Same person should be 1 instance"

def test_different_objects_counted_separately():
    """Test that different objects count separately"""
    tracker = ObjectInstanceTracker(iou_threshold=0.3)

    # Two persons at different locations
    detections = [
        [
            {'class': 'person', 'bbox': [100, 100, 200, 300], 'timestamp': 0.0},
            {'class': 'person', 'bbox': [400, 100, 500, 300], 'timestamp': 0.0}
        ],
    ]

    instances = tracker.track_objects(detections)
    assert len(instances) == 2, "Different persons should be 2 instances"

def test_reappearing_object():
    """Test object that disappears and reappears"""
    tracker = ObjectInstanceTracker(iou_threshold=0.3, max_frames_missing=2)

    # Object appears, disappears, reappears within threshold
    detections = [
        [{'class': 'cup', 'bbox': [100, 100, 150, 150], 'timestamp': 0.0}],
        [],  # Missing frame
        [],  # Missing frame
        [{'class': 'cup', 'bbox': [105, 100, 155, 150], 'timestamp': 1.0}],
    ]

    instances = tracker.track_objects(detections)
    assert len(instances) == 1, "Reappearing object within threshold should be same instance"
```

## Edge Cases to Handle

1. **Fast Moving Objects**: Lower IoU threshold (0.3) to accommodate movement
2. **Occlusion**: Track objects that temporarily disappear (max_frames_missing)
3. **Multiple Same Class**: Properly distinguish multiple persons/bottles
4. **Scene Changes**: Reset tracking at scene boundaries if needed
5. **Confidence Threshold**: Only track objects with confidence > 0.3

## Expected Impact

### Before Fix
```json
{
  "hook": {
    "object_count": 17,  // 10 person + 6 bottle + 1 cup detections
    "person_count": 1
  }
}
```

### After Fix
```json
{
  "hook": {
    "object_count": 3,   // 1 person + 1 bottle + 1 cup (unique instances)
    "person_count": 1
  }
}
```

## Validation Strategy

1. Run on test video with known object counts
2. Verify hook/middle/closing windows have reasonable counts
3. Check that person_count ≤ object_count
4. Ensure object_count reflects actual unique items shown

## Alternative Considerations

### Simpler Approach (Not Recommended)
Count unique object classes: `object_count = len(set(obj['class'] for obj in segment_objects))`
- Pros: Simple, no tracking needed
- Cons: Can't distinguish multiple instances of same class

### Complex Approach (Future Enhancement)
Use deep learning re-identification models
- Pros: More accurate for person tracking
- Cons: Computationally expensive, overkill for this use case

## Implementation Order

1. Create ObjectInstanceTracker class with tests
2. Integrate into temporal_compute.py
3. Validate on sample videos
4. Update any downstream dependencies expecting old counts
5. Document the change in release notes

## Notes

- IoU threshold of 0.3 chosen to balance movement tolerance vs false merging
- Max frames missing of 5 (≈1.5 seconds at 3fps) handles brief occlusions
- This approach works well for stationary camera TikTok videos
- May need adjustment for videos with camera movement