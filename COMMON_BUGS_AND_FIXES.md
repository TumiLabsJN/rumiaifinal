# Common Bugs and Fixes in RumiAI v2

## Table of Contents
1. [Function Signature Mismatches](#function-signature-mismatches)
2. [JSON Response Format Issues](#json-response-format-issues)
3. [Naming Inconsistencies](#naming-inconsistencies)
4. [Undefined Variables](#undefined-variables)
5. [Data Structure Mismatches](#data-structure-mismatches)
6. [Import and Dependency Issues](#import-and-dependency-issues)
7. [Template and Prompt Issues](#template-and-prompt-issues)
8. [Validation Logic Errors](#validation-logic-errors)

## 1. Function Signature Mismatches

### Problem
Functions expect arguments in a specific order, but callers pass them in different order or miss arguments entirely.

### Examples
```python
# Function expects:
def compute_speech_analysis_metrics(speech_timeline, transcript, speech_segments, 
                                   expression_timeline, gesture_timeline, 
                                   human_analysis_data, video_duration):

# But was called with:
compute_speech_analysis_metrics(transcript, speech_segments, expression_timeline, 
                               gesture_timeline, human_analysis_data, video_duration)
# Missing speech_timeline as first argument!
```

### Fixes
1. **Create wrapper functions** that extract and pass arguments in correct order:
```python
def compute_speech_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for speech analysis computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    speech_timeline = timelines.get('speechTimeline', {})
    
    # ... extract other required data ...
    
    return compute_speech_analysis_metrics(
        speech_timeline, transcript, speech_segments, expression_timeline, 
        gesture_timeline, human_analysis_data, video_duration
    )
```

2. **Use keyword arguments** to avoid order issues:
```python
return compute_speech_analysis_metrics(
    speech_timeline=speech_timeline,
    transcript=transcript,
    speech_segments=speech_segments,
    # ... etc
)
```

## 2. JSON Response Format Issues

### Problem
Claude returns responses in various formats (markdown, plain text, JSON with different structures) instead of expected pure JSON.

### Examples
```python
# Expected JSON:
{
  "densityCoreMetrics": { ... },
  "densityDynamics": { ... }
}

# But Claude returns:
"Based on the analysis, here are the results:\n\n```json\n{\n  \"densityCoreMetrics\": ...\n}\n```"
```

### Fixes
1. **Update prompt templates** with explicit instructions:
```text
CRITICAL: Output ONLY valid JSON with these exact 6 blocks as a single JSON object:
{
  "densityCoreMetrics": { ... },
  "densityDynamics": { ... },
  "densityInteractions": { ... },
  "densityKeyEvents": { ... },
  "densityPatterns": { ... },
  "densityQuality": { ... }
}

Do NOT include any markdown formatting, explanations, or text outside the JSON structure.
```

2. **Add fallback parsing** in validator:
```python
def extract_json_from_markdown(response: str) -> Optional[Dict]:
    """Extract JSON from markdown code blocks"""
    import re
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None
```

## 3. Naming Inconsistencies

### Problem
Different parts of the system use different names for the same concept.

### Examples
```python
# Prompt type in runner:
'visual_overlay_analysis'

# Template file name:
'visual_overlay_v2.txt'

# Function name:
'compute_visual_overlay_metrics'

# Block names in response:
'overlaysCoreMetrics'
```

### Fixes
1. **Create mapping dictionaries**:
```python
BLOCK_NAME_MAPPINGS = {
    'creative_density': {
        'densityCoreMetrics': 'CoreMetrics',
        'densityDynamics': 'Dynamics',
        # ... etc
    },
    'visual_overlay_analysis': {
        'overlaysCoreMetrics': 'CoreMetrics',
        'overlaysDynamics': 'Dynamics',
        # ... etc
    }
}
```

2. **Standardize naming conventions**:
- Use consistent suffixes (_analysis, _metrics, etc.)
- Document naming patterns
- Create constants for reused names

## 4. Undefined Variables

### Problem
Variables used without being defined in scope, often due to copy-paste errors or missing imports.

### Examples
```python
# In compute_person_framing_metrics:
if scene_change_timeline:  # Error: scene_change_timeline not defined
    for timestamp, scene_data in scene_change_timeline.items():
        # ...

# Also:
authenticity_factors = [
    energy_variance > 0.1,  # Error: energy_variance not defined
]
```

### Fixes
1. **Remove or replace undefined references**:
```python
# Option 1: Remove the check
shot_transitions = []
# scene_change_timeline not available, detect transitions differently

# Option 2: Provide default/alternative
energy_variance = 0.5  # Default value
# Or remove from calculation entirely
```

2. **Add proper imports and definitions**:
```python
# At module level
scene_change_timeline = {}  # Initialize if needed

# Or pass as parameter
def compute_metrics(..., scene_change_timeline=None):
    if scene_change_timeline:
        # use it
```

## 5. Data Structure Mismatches

### Problem
Code expects data in one format but receives it in another.

### Examples
```python
# Expected timeline entry:
entry.get('start', {}).get('seconds', 0)  # Expects nested dict

# Actual timeline entry:
{
    "start": "3s",  # String, not dict!
    "entry_type": "scene_change"
}
```

### Fixes
1. **Add type checking and conversion**:
```python
# Extract seconds from various formats
start_str = entry.get('start', '0s')
if isinstance(start_str, str) and start_str.endswith('s'):
    start_seconds = int(start_str[:-1])
elif isinstance(start_str, dict):
    start_seconds = start_str.get('seconds', 0)
else:
    start_seconds = 0
```

2. **Create data normalization functions**:
```python
def normalize_timestamp(timestamp):
    """Convert various timestamp formats to seconds"""
    if isinstance(timestamp, str):
        return parse_timestamp_string(timestamp)
    elif isinstance(timestamp, dict):
        return timestamp.get('seconds', 0)
    return float(timestamp)
```

## 6. Import and Dependency Issues

### Problem
Circular imports, missing modules, or incorrect import paths.

### Examples
```python
# Circular import:
# In processors/__init__.py
from .precompute_functions import get_compute_function

# In precompute_functions.py
from . import some_processor  # Circular!

# Missing module:
from .precompute_functions_full import compute_creative_density_analysis
# FileNotFoundError if file doesn't exist
```

### Fixes
1. **Use lazy imports**:
```python
def get_compute_function(name: str):
    # Import only when needed
    from .precompute_functions_full import compute_creative_density_analysis
    return compute_creative_density_analysis
```

2. **Provide fallbacks for missing imports**:
```python
try:
    from .precompute_functions_full import compute_creative_density_analysis
except ImportError as e:
    logger.error(f"Failed to import: {e}")
    # Define placeholder
    def compute_creative_density_analysis(*args, **kwargs):
        logger.warning("Using placeholder function")
        return {}
```

## 7. Template and Prompt Issues

### Problem
Prompt templates don't match what the code expects or don't produce desired output format.

### Examples
```python
# Template instructs to output generic blocks:
"Output the following 6 modular blocks:"

# But validator expects specific names:
"densityCoreMetrics", "densityDynamics", etc.
```

### Fixes
1. **Synchronize templates with validators**:
```python
# In template:
"Output these exact blocks: densityCoreMetrics, densityDynamics..."

# In validator:
EXPECTED_BLOCKS = ['densityCoreMetrics', 'densityDynamics', ...]
```

2. **Add explicit examples in templates**:
```text
Example output structure:
{
  "densityCoreMetrics": {
    "avgDensity": 2.5,
    "maxDensity": 5,
    ...
  }
}
```

## 8. Validation Logic Errors

### Problem
Validators reject valid responses due to overly strict or incorrect validation logic.

### Examples
```python
# Validator rejects response because it's checking for generic names
# but response has prompt-specific names
if block_name not in cls.EXPECTED_BLOCKS:
    errors.append(f"Unexpected block: {block_name}")
```

### Fixes
1. **Add context-aware validation**:
```python
# Detect which naming scheme is used
prompt_type_detected = None
for pt, mapping in cls.BLOCK_NAME_MAPPINGS.items():
    if any(block_name in found_blocks for block_name in mapping.keys()):
        prompt_type_detected = pt
        break

# Skip strict validation for prompt-specific formats
if prompt_type_detected:
    # Use mapping to normalize names
    pass
```

2. **Make validators more flexible**:
```python
# Allow either generic or specific names
ALLOWED_BLOCKS = set(EXPECTED_BLOCKS)
for mapping in BLOCK_NAME_MAPPINGS.values():
    ALLOWED_BLOCKS.update(mapping.keys())
```

## Best Practices to Avoid These Bugs

### 1. **Type Hints and Documentation**
```python
def process_timeline(
    timeline_data: Dict[str, Any],
    duration: float
) -> Dict[str, List[float]]:
    """
    Process timeline data.
    
    Args:
        timeline_data: Timeline with entries like {"start": "0s", ...}
        duration: Video duration in seconds
        
    Returns:
        Processed metrics dictionary
    """
```

### 2. **Defensive Programming**
```python
# Always provide defaults
value = data.get('key', default_value)

# Check types before using
if isinstance(value, str):
    # handle string
elif isinstance(value, dict):
    # handle dict
```

### 3. **Unit Tests**
```python
def test_compute_function_with_missing_data():
    """Test that functions handle missing data gracefully"""
    result = compute_metrics({})  # Empty input
    assert result is not None
    assert 'error' not in result
```

### 4. **Consistent Naming Convention**
- Use a naming guide document
- Stick to patterns like `{feature}_{type}_{version}`
- Example: `creative_density_analysis_v2.txt`

### 5. **Error Logging**
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    # Provide fallback
    result = default_result()
```

### 6. **Integration Testing**
- Test the full pipeline with various inputs
- Include edge cases (empty data, malformed data)
- Verify all components work together

## Debugging Tips

1. **Add debug logging at boundaries**:
```python
logger.debug(f"Calling function with args: {args}")
result = function(*args)
logger.debug(f"Function returned: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
```

2. **Validate data at entry points**:
```python
def process_data(data):
    # Validate input
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert 'required_key' in data, "Missing required_key"
    
    # Process...
```

3. **Use descriptive error messages**:
```python
if not timeline_entries:
    raise ValueError(
        f"No timeline entries found. Expected format: "
        f"[{{'start': '0s', 'entry_type': 'scene', ...}}, ...]"
    )
```

## Conclusion

Most bugs fall into these categories and can be prevented with:
- Careful attention to data structures and types
- Comprehensive error handling
- Clear documentation and examples
- Consistent naming conventions
- Thorough testing at both unit and integration levels

When debugging, always check:
1. What format is the data actually in? (print/log it)
2. What format does the code expect? (read the function)
3. Are all variables defined? (check imports and scope)
4. Do names match across components? (templates, functions, validators)