# Emotion Service Integration Manual - Architectural Lessons Learned

## Overview
This document captures critical learnings from integrating FEAT (Facial Expression Analysis Toolkit) emotion detection into the RumiAI pipeline. It serves as a guide for future ML service integrations and highlights architectural patterns that must be respected.

**Purpose**: Enable a fresh Claude instance to understand the architecture and avoid common pitfalls when integrating new ML services.

## Core Architecture Understanding

### 1. The Three-Layer Data Flow Pattern

RumiAI follows a strict three-layer pattern for ML data:

```
ML Service → Timeline Builder → Timeline Entries → Precompute Functions
     ↓              ↓                    ↓                    ↓
  Raw Output   Validation &        Unified          Extracted for
               Transformation       Storage          Analysis
```

**CRITICAL**: Never bypass this flow. Services like `audio_energy` that bypass timeline_builder create inconsistencies and technical debt.

### 2. Service Integration Checklist

When integrating a new ML service, you MUST touch these files in order:

1. **`/rumiai_v2/ml_services/[service_name]_service.py`**
   - Implements the actual ML processing
   - Outputs to service-specific directory
   - Maps service output to RumiAI's expected format

2. **`/rumiai_v2/core/validators/ml_data_validator.py`**
   - Add `validate_[service]_data()` method
   - FAIL-FAST principle: Raise exceptions, don't normalize bad data
   - Trust the validator downstream (no redundant checks)

3. **`/rumiai_v2/processors/timeline_builder.py`**
   - Add service to `builders` dictionary
   - Implement `_add_[service]_entries()` method
   - Transform validated data into TimelineEntry objects

4. **`/rumiai_v2/processors/precompute_functions.py`**
   - Extract timeline entries to specialized timelines
   - Build service-specific timeline dictionaries
   - NO direct ml_data access (that's an anti-pattern)

## Critical Architectural Principles

### 1. Fail-Fast Philosophy

```python
# ✅ CORRECT - Fail immediately on bad data
if 'required_field' not in data:
    raise ValueError(f"FATAL: Missing required_field for video {video_id}")

# ❌ WRONG - Silent normalization
if 'required_field' not in data:
    data['required_field'] = 'default_value'  # Creates hidden bugs
```

**Why**: Silent failures cascade into mysterious bugs. Better to crash early with clear error messages.

### 2. Trust the Validator Pattern

```python
# In timeline_builder.py after validation:
timestamp = emotion_entry['timestamp']  # ✅ No .get(), no default

# NOT:
timestamp = emotion_entry.get('timestamp', 0)  # ❌ Redundant defensive coding
```

**Why**: If validator passes bad data, that's a validator bug. Don't hide it with defaults.

### 3. Timeline Entry Structure

TimelineEntry has exactly these fields:
- `start`: Timestamp object
- `end`: Optional[Timestamp] 
- `entry_type`: str
- `data`: Dict[str, Any]

**COMMON BUG**: Adding extra parameters
```python
# ❌ WRONG - This will crash
entry = TimelineEntry(
    entry_type='emotion',
    confidence=0.9,  # NO! TimelineEntry doesn't have this
    metadata={...}   # NO! This doesn't exist either
)

# ✅ CORRECT - Everything goes in data dict
entry = TimelineEntry(
    entry_type='emotion',
    start=Timestamp(timestamp),
    end=Timestamp(timestamp + 1),
    data={
        'confidence': 0.9,
        'metadata': {...}
    }
)
```

### 4. Service Name Consistency

The service name must be IDENTICAL across all layers:
- ML service output key: `'emotion_detection'`
- Timeline builder key: `'emotion_detection': self._add_emotion_entries`
- Entry type: `'emotion'` (can be different, but document it)

**GOTCHA**: FEAT outputs 'happiness' but RumiAI uses 'joy'. Map this in the service layer, not downstream.

## Common Integration Pitfalls

### Pitfall 1: Checking Wrong Documentation

**Problem**: Line 16 of FEATFIX.md said "7 basic emotions (anger, disgust, fear, happiness...)" but the actual JSON output had "joy" not "happiness".

**Solution**: ALWAYS check actual output files:
```bash
# Don't trust documentation, verify actual output
cat emotion_detection_outputs/*/[video_id]_emotions.json | head -50
```

### Pitfall 2: Duplicate Service Patterns

**Problem**: Audio energy bypasses timeline_builder, going directly from ml_results to precompute.

**Why This Is Bad**:
- Inconsistent with other services
- Hidden dependency in precompute_functions.py
- Can't query audio energy from timeline
- Testing is more complex

**Solution**: Follow the standard pattern even if it seems like overhead.

### Pitfall 3: Import Path Assumptions

**Problem**: Assumed relative imports without checking existing patterns.

**Solution**: Check how timeline_builder.py already imports:
```python
# timeline_builder.py already has:
from ..core.validators import MLDataValidator

# So use the existing instance:
self.ml_validator.validate_emotion_data(...)  # ✅

# Not:
from ..core.validators import MLDataValidator  # ❌ Redundant
```

### Pitfall 4: MediaPipe Expression Confusion

**Problem**: MediaPipe also populates expressionTimeline with basic face data.

**Solution**: Services should be prioritized:
1. Check if better service (FEAT) has data
2. Only fall back to MediaPipe if FEAT unavailable
3. Mark data source: `'source': 'feat'` vs `'source': 'mediapipe'`

## Testing Integration

### Essential Test Coverage

1. **Validator Test**: Does it reject bad data?
```python
# Should raise ValueError
bad_data = {'emotions': [{'timestamp': -1}]}  # Invalid timestamp
validator.validate_emotion_data(bad_data)
```

2. **Timeline Builder Test**: Does it create proper entries?
```python
timeline = Timeline(video_id='test', duration=60)
builder._add_emotion_entries(timeline, feat_data)
emotion_entries = [e for e in timeline.entries if e.entry_type == 'emotion']
assert emotion_entries[0].data['source'] == 'feat'
```

3. **Extraction Test**: Does it populate the right timeline?
```python
timelines = _extract_timelines_from_analysis(analysis_dict)
assert 'feat' in timelines['expressionTimeline']['0-1s'].get('source')
```

### Integration Test Pattern

Always test the full flow:
```bash
# Real video with existing service output
python3 scripts/rumiai_runner.py "test_video.mp4"
grep "expressionTimeline" insights/*/emotional_journey/*.json
```

## Debugging Workflow

When integration fails, check in this order:

1. **Is the service running?**
   ```bash
   ls -la [service]_outputs/
   ```

2. **Is it in video_analyzer.py?**
   ```bash
   grep -n "emotion_detection" rumiai_v2/api/video_analyzer.py
   ```

3. **Is it in timeline_builder.py builders dict?**
   ```bash
   grep -n "builders = {" rumiai_v2/processors/timeline_builder.py -A 10
   ```

4. **Is the validator being called?**
   ```bash
   grep -n "validate_emotion_data" rumiai_v2/processors/timeline_builder.py
   ```

5. **Is extraction happening?**
   ```bash
   grep -n "entry_type.*emotion" rumiai_v2/processors/precompute_functions.py
   ```

## Service Discovery Process

To understand how a service should integrate:

1. **Find existing examples**:
   ```bash
   # See how YOLO is integrated (it's a good pattern)
   grep -r "_add_yolo_entries" .
   ```

2. **Check the data flow**:
   ```bash
   # Trace from service to timeline
   grep -r "yolo" --include="*.py" | grep -E "(timeline|Timeline)"
   ```

3. **Verify the models**:
   ```bash
   # Understand TimelineEntry structure
   grep "class TimelineEntry" -A 20 rumiai_v2/core/models/timeline.py
   ```

## Architectural Invariants

These must NEVER be violated:

1. **All ML services go through timeline_builder.py**
   - Exception: None should exist

2. **Validators fail fast with exceptions**
   - Never return normalized/default data

3. **Timeline entries use only defined fields**
   - start, end, entry_type, data

4. **Service names are consistent across layers**
   - Same key in ml_results, builders dict

5. **Data source is marked when multiple services provide similar data**
   - e.g., 'source': 'feat' vs 'source': 'mediapipe'

## Quick Reference: Adding New Service

```bash
# 1. Check if service already runs
ls -la ml_outputs/ | grep [service]
ls -la [service]_outputs/

# 2. Find where services are listed
grep "builders = {" timeline_builder.py -A 10

# 3. Add to builders dict
'new_service': self._add_new_service_entries,

# 4. Implement method (copy pattern from _add_yolo_entries)
def _add_new_service_entries(self, timeline: Timeline, service_data: Dict[str, Any]) -> None:
    # Validate
    service_data = self.ml_validator.validate_new_service_data(service_data, timeline.video_id)
    
    # Transform to entries
    for item in service_data.get('items', []):
        entry = TimelineEntry(
            entry_type='new_type',
            start=Timestamp(item['timestamp']),
            end=None,  # or Timestamp(item['timestamp'] + duration)
            data={...}
        )
        timeline.add_entry(entry)

# 5. Add validator
@staticmethod
def validate_new_service_data(data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]:
    if 'required_field' not in data:
        raise ValueError(f"FATAL: Missing required_field for video {video_id}")
    return data

# 6. Extract in precompute_functions.py
for entry in timeline_entries:
    if entry.get('entry_type') == 'new_type':
        # Extract and add to appropriate timeline
```

## Final Wisdom

1. **Read the actual code, not just documentation**
2. **Check actual output files, not assumed formats**
3. **Follow existing patterns, don't create new ones**
4. **Fail fast and loud, never silent**
5. **Test with real data, not just mocks**
6. **When in doubt, grep the codebase for examples**

Remember: The architecture exists for a reason. Respect it, and it will respect you back with fewer bugs and easier maintenance.