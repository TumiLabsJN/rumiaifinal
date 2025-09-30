# Speech Pitch Fix: Removing avg_pitch_normalized

## Executive Decision

**Decision**: Hard-remove `avg_pitch_normalized` feature from the temporal compute pipeline
**Rationale**: Feature contains systematic harmonic detection error that makes it misleading rather than useful
**Gender Detection**: PRESERVED in metadata for other uses
**Date**: 2025-09-30

## Problem Statement

The `avg_pitch_normalized` feature suffers from a fundamental flaw where librosa.piptrack frequently detects the 2nd harmonic instead of the fundamental frequency. This causes:

1. Male voices measured at ~288Hz instead of ~144Hz (2x error)
2. Normalized values capped at 3.0 (maximum) for most male speakers
3. Inconsistent measurements across different voice/microphone combinations
4. Complete failure of gender-relative normalization logic

## Why Remove Instead of Fix

1. **Poor ROI**: 1-4 weeks to implement robust pitch detection for a redundant feature
2. **Overlapping Coverage**: `pitch_scatter_ratio` already captures vocal dynamics
3. **Validation Complexity**: Would need extensive testing across diverse voices
4. **Maintenance Burden**: Pitch detection is inherently fragile
5. **avg_pitch_hz also flawed**: Suffers from same harmonic detection error (288Hz vs 144Hz)

## Implementation Plan

### 1. Core Code Changes (temporal_compute.py)

#### Remove from calculate_pitch_metrics() function

**BEFORE:**
```python
def calculate_pitch_metrics(pitch_timeline, video_duration, start_time, end_time, gender):
    # ... voiced pitch extraction ...

    # Gender-based normalization
    if gender == 'male':
        baseline_hz = 110
        range_hz = 40
    elif gender == 'female':
        baseline_hz = 200
        range_hz = 45
    else:  # multiple_people
        if len(voiced_pitches) >= 20:
            baseline_hz = np.percentile(voiced_pitches, 20)
            range_hz = np.percentile(voiced_pitches, 80) - baseline_hz
        else:
            baseline_hz = 150
            range_hz = 100

    avg_pitch_normalized = (avg_pitch_hz - baseline_hz) / range_hz
    avg_pitch_normalized = max(-1.0, min(3.0, avg_pitch_normalized))

    pitch_scatter_ratio = (max(voiced_pitches) - min(voiced_pitches)) / avg_pitch_hz

    return avg_pitch_hz, avg_pitch_normalized, pitch_scatter_ratio
```

**AFTER:**
```python
def calculate_pitch_metrics(pitch_timeline, video_duration, start_time, end_time):
    # Note: Removed gender parameter - no longer needed
    # avg_pitch_normalized removed due to harmonic detection error - see SpeechPitchFix.md
    # avg_pitch_hz also removed - only used internally, never stored

    # ... voiced pitch extraction ...

    # Early return if insufficient voiced content
    if len(voiced_pitches) < 10:
        return None  # No pitch metrics available

    avg_pitch_hz = float(np.mean(voiced_pitches))

    # Safety check (shouldn't happen with voiced pitches)
    if avg_pitch_hz == 0:
        return None

    # Only calculate and return pitch_scatter_ratio
    pitch_scatter_ratio = (max(voiced_pitches) - min(voiced_pitches)) / avg_pitch_hz

    return pitch_scatter_ratio  # Returns single value or None
```

**Note on pitch_scatter_ratio usefulness**: This metric remains valuable on its own as it measures vocal control/instability independent of absolute pitch. High values indicate scattered/unstable pitch (whisper, nervousness), while low values indicate controlled delivery, regardless of the speaker's vocal range.

#### Update process_temporal_windows() function

**BEFORE:**
```python
# In the window processing loop
if pitch_timeline:
    avg_pitch_hz, avg_pitch_normalized, pitch_scatter_ratio = calculate_pitch_metrics(
        pitch_timeline, video_duration, window_start, window_end, gender
    )
    window_features['avg_pitch_normalized'] = avg_pitch_normalized
    window_features['pitch_scatter_ratio'] = pitch_scatter_ratio
```

**AFTER:**
```python
# In the window processing loop
if pitch_timeline:
    pitch_scatter_ratio = calculate_pitch_metrics(
        pitch_timeline, video_duration, window_start, window_end
    )  # Returns single value or None

    # Handle case where pitch metrics unavailable
    if pitch_scatter_ratio is not None:
        window_features['pitch_scatter_ratio'] = pitch_scatter_ratio
    else:
        # No pitch metrics available for this window
        window_features['pitch_scatter_ratio'] = 0.0  # 0.0 means "no voiced content detected"
```

### 2. Verify and PRESERVE Gender Detection

#### First, verify gender is ONLY used for two purposes:
```bash
# Search for all gender usage in Python files
grep -r "gender" --include="*.py" rumiai_v2/ | grep -v "avg_pitch_normalized"

# Also check configuration and data files
grep -r "gender" --include="*.json" rumiai_v2/
grep -r "gender" --include="*.yaml" rumiai_v2/
grep -r "gender" --include="*.config" rumiai_v2/

# Expected findings:
# 1. Gender detection code itself (deepface)
# 2. Metadata assignment
# 3. No other feature dependencies in any configuration
```

**After verification, NO CHANGES to these sections:**
```python
# This code stays exactly as is:
calculated_metadata['gender_detection'] = {
    'gender': gender,
    'confidence': confidence,
    'method': 'deepface'
}
```

Gender detection remains in the metadata section of the JSON output for:
- Future feature development
- Creator demographics analysis
- Any other non-pitch use cases

### 3. Identify ALL Function Call Sites

Before making changes, locate every call to `calculate_pitch_metrics`:
```bash
# Find all calls to the function that needs signature change
grep -rn "calculate_pitch_metrics" --include="*.py" rumiai_v2/

# Document each finding:
# rumiai_v2/processors/temporal_compute.py:1150 (main processing loop)
# rumiai_v2/processors/temporal_compute.py:1195 (closing window)
# [Add any other occurrences found]
```

Each call site must be updated to:
1. Remove the gender parameter
2. Handle the new (None, None) return possibility

### 4. Documentation Updates

#### TotalFeatures.md
Remove line 21:
```markdown
| avg_pitch_normalized | Pitch | Audio Energy, DeepFace | gender_detection for normalization | Temporal | Float [-1-3] | ...
```

#### AudioFeatures.md
Remove:
- Lines 27-28: Feature table entry
- Lines 244-249: JSON example with avg_pitch_normalized
- Lines 258-259: Metric definition for avg_pitch_normalized
- Sections explaining gender normalization for pitch

#### AudioServices.md
Update JSON examples to show pitch_scatter_ratio but not avg_pitch_normalized

### 5. Test File Updates

Remove old validation checks and ADD new tests:
```python
# DELETE these assertions from existing test files:
assert 'avg_pitch_normalized' in data['temporal_windows']['hook']
assert -1 <= window_data['avg_pitch_normalized'] <= 3

# KEEP this assertion:
assert 'gender_detection' in data['metadata']  # Gender stays!
```

#### Add New Unit Tests (test_pitch_removal.py)
```python
import json
import unittest
from rumiai_v2.processors.temporal_compute import process_temporal_windows

class TestPitchRemoval(unittest.TestCase):

    def test_no_avg_pitch_normalized_in_output(self):
        """Verify avg_pitch_normalized is not in any temporal window"""
        result = process_temporal_windows('test_video.mp4')

        # Check hook
        self.assertNotIn('avg_pitch_normalized', result['temporal_windows']['hook'])

        # Check middle segments
        for segment in result['temporal_windows'].get('middle_segments', []):
            self.assertNotIn('avg_pitch_normalized', segment)

        # Check closing
        self.assertNotIn('avg_pitch_normalized', result['temporal_windows']['closing'])

    def test_gender_detection_still_present(self):
        """Verify gender detection remains in metadata"""
        result = process_temporal_windows('test_video.mp4')

        self.assertIn('gender_detection', result['metadata'])
        gender_data = result['metadata']['gender_detection']
        self.assertIn('gender', gender_data)
        self.assertIn('confidence', gender_data)
        self.assertIn('method', gender_data)

    def test_pitch_scatter_ratio_handles_no_voice(self):
        """Verify pitch_scatter_ratio handles videos with no voiced content"""
        result = process_temporal_windows('silent_video.mp4')

        # Should either be 0.0 or omitted
        pitch_val = result['temporal_windows']['hook'].get('pitch_scatter_ratio', 0.0)
        self.assertEqual(pitch_val, 0.0)

    def test_calculate_pitch_metrics_signature(self):
        """Verify function signature no longer accepts gender parameter"""
        from rumiai_v2.processors.temporal_compute import calculate_pitch_metrics
        import inspect

        sig = inspect.signature(calculate_pitch_metrics)
        param_names = list(sig.parameters.keys())

        # Verify gender parameter is not present
        self.assertNotIn('gender', param_names)

    def test_calculate_pitch_metrics_return_value(self):
        """Verify function returns single value or None"""
        from rumiai_v2.processors.temporal_compute import calculate_pitch_metrics

        # Mock data with sufficient voiced content
        result = calculate_pitch_metrics(mock_pitch_timeline, 10.0, 0.0, 3.0)

        # Should be either a float or None, not a tuple
        self.assertTrue(result is None or isinstance(result, float))

        # If not None, should be between 0 and 1 (ratio)
        if result is not None:
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
```

### 6. Execution Steps

```bash
# 1. Pre-change verification
# Identify all files that need changes
grep -rn "calculate_pitch_metrics" --include="*.py" rumiai_v2/ > call_sites.txt
grep -r "gender" --include="*.py" rumiai_v2/ | grep -v "avg_pitch_normalized" > gender_usage.txt
grep -r "gender" --include="*.json" --include="*.yaml" --include="*.config" rumiai_v2/ > gender_config.txt

# 2. Create test file first
cat > test_pitch_removal.py << 'EOF'
import json
import unittest
from rumiai_v2.processors.temporal_compute import process_temporal_windows, calculate_pitch_metrics
import inspect

class TestPitchRemoval(unittest.TestCase):

    def test_no_avg_pitch_normalized_in_output(self):
        """Verify avg_pitch_normalized is not in any temporal window"""
        result = process_temporal_windows('test_video.mp4')

        self.assertNotIn('avg_pitch_normalized', result['temporal_windows']['hook'])

        for segment in result['temporal_windows'].get('middle_segments', []):
            self.assertNotIn('avg_pitch_normalized', segment)

        self.assertNotIn('avg_pitch_normalized', result['temporal_windows']['closing'])

    def test_gender_detection_still_present(self):
        """Verify gender detection remains in metadata"""
        result = process_temporal_windows('test_video.mp4')

        self.assertIn('gender_detection', result['metadata'])
        gender_data = result['metadata']['gender_detection']
        self.assertIn('gender', gender_data)
        self.assertIn('confidence', gender_data)
        self.assertIn('method', gender_data)

    def test_pitch_scatter_ratio_handles_no_voice(self):
        """Verify pitch_scatter_ratio handles videos with no voiced content"""
        result = process_temporal_windows('silent_video.mp4')

        pitch_val = result['temporal_windows']['hook'].get('pitch_scatter_ratio', 0.0)
        self.assertEqual(pitch_val, 0.0)

    def test_calculate_pitch_metrics_signature(self):
        """Verify function signature no longer accepts gender parameter"""
        sig = inspect.signature(calculate_pitch_metrics)
        param_names = list(sig.parameters.keys())

        self.assertNotIn('gender', param_names)

    def test_calculate_pitch_metrics_return_value(self):
        """Verify function returns single value or None"""
        # This test would need proper mock data to run
        pass

if __name__ == '__main__':
    unittest.main()
EOF

# 3. Backup current state
cp rumiai_v2/processors/temporal_compute.py rumiai_v2/processors/temporal_compute.py.backup_pitch_removal

# 4. Edit temporal_compute.py
# - Remove avg_pitch_normalized calculation
# - Add early return for None case in calculate_pitch_metrics
# - Remove gender parameter from function signature
# - Update ALL call sites identified in call_sites.txt
# - Add None handling at each call site

# 5. Run unit tests
python3 -m pytest test_pitch_removal.py -v

# 6. Test with sample videos
python3 test_manual_videos.py Video08GenderMale.mp4
# Verify JSON has no avg_pitch_normalized but STILL has gender_detection

# 7. Update documentation files
# - TotalFeatures.md (remove line 21)
# - AudioFeatures.md (remove lines 27-28, 244-249, 258-259)
# - AudioServices.md (update JSON examples)

# 8. Run full test suite
python3 -m pytest rumiai_v2/tests/ -v

# 9. Final verification
# Confirm no avg_pitch_normalized in output
python3 -c "import json; d=json.load(open('insights/latest_output.json')); assert 'avg_pitch_normalized' not in str(d)"

# Confirm gender_detection still present
python3 -c "import json; d=json.load(open('insights/latest_output.json')); assert d['metadata']['gender_detection']"
```

## Verification Checklist

- [ ] `avg_pitch_normalized` removed from temporal_compute.py
- [ ] `calculate_pitch_metrics()` no longer takes gender parameter
- [ ] `calculate_pitch_metrics()` returns single value or None
- [ ] All calls to `calculate_pitch_metrics()` updated
- [ ] No `avg_pitch_normalized` in new JSON outputs
- [ ] **gender_detection STILL in metadata section** ✓
- [ ] Documentation updated in 3 files
- [ ] Tests pass without avg_pitch_normalized checks

## Default Value Documentation

**Important**: When `pitch_scatter_ratio` = 0.0, this indicates "no voiced content detected" rather than "perfectly stable pitch". This is documented as the standard sentinel value for missing pitch data throughout the system.

## Expected JSON Output After Fix

```json
{
  "temporal_windows": {
    "hook": {
      // No avg_pitch_normalized here
      "pitch_scatter_ratio": 0.619,  // or 0.0 if no voiced content
      "energy_level": 0.046,
      // ... other features ...
    }
  },
  "metadata": {
    "gender_detection": {  // THIS STAYS!
      "gender": "male",
      "confidence": 0.9479,
      "method": "deepface"
    },
    // ... other metadata ...
  }
}
```

## Benefits

1. **Removes misleading data** - No more 2x frequency errors in both avg_pitch_normalized and avg_pitch_hz
2. **Simplifies pipeline** - Function now returns single clear value
3. **Preserves gender data** - Available for other features
4. **Cleaner ML training** - No capped/broken features
5. **Cleaner interface** - No unused return values (avg_pitch_hz was never stored)

## Rollback Plan

If issues arise:
```bash
# Restore from backup
cp rumiai_v2/processors/temporal_compute.py.backup_pitch_removal rumiai_v2/processors/temporal_compute.py
```

## Notes

- This is a breaking change - old JSON files will have orphaned `avg_pitch_normalized` fields
- New processing will not include the field at all (hard remove)
- Gender detection continues to run and appears in metadata
- Consider adding comment in code: `# avg_pitch_normalized removed due to harmonic detection error - see SpeechPitchFix.md`