# Test Flow v3: Enhanced ML Service Validation Strategy

## Problem Statement
`test_temporal_compute_v2.py` claims to be a "true mirror" of production but has critical gaps in ML service validation, making it unreliable for testing temporal compute accuracy.

## Key Issues Identified

### 1. Missing ML Service Validation
- Test assumes all 8 ML services are present in unified_analysis
- No verification that services (emotion_detection, deepface_gender) actually ran
- Could test with incomplete data without knowing

### 2. Incomplete Output Comparison
- Only spot-checks 9 values instead of comparing complete output
- Ignores middle_segments and closing windows entirely
- Doesn't verify JSON equality with production


## Proposed Solution Architecture

### Phase 1: ML Service Validation (Within test_temporal_compute_v2.py)

#### Add `validate_ml_services()` function after `load_unified_analysis()`:

```python
def validate_ml_services(unified_dict: Dict[str, Any]) -> Dict[str, str]:
    """
    Validates presence and quality of all ML services.
    Returns status dict with 'present'/'missing'/'empty' for each service.
    """
    REQUIRED_SERVICES = {
        # Timeline-based services (go through timeline builder)
        'yolo': {'required_fields': ['objectAnnotations']},
        'whisper': {'required_fields': ['segments']},
        'mediapipe': {'required_fields': ['poses', 'faces']},
        'ocr': {'required_fields': ['textAnnotations']},
        'scene_detection': {'required_fields': ['scenes']},
        'emotion_detection': {'required_fields': ['frames']},

        # Direct ML data services (bypass timeline)
        'audio_energy': {'required_fields': ['rms_frames']},
        'deepface_gender': {'required_fields': ['gender', 'confidence']}
    }

    ml_data = unified_dict.get('ml_data', {})
    validation_status = {}

    for service, config in REQUIRED_SERVICES.items():
        if service not in ml_data:
            validation_status[service] = 'missing'
        elif not ml_data[service]:
            validation_status[service] = 'empty'
        else:
            # Check required fields
            has_fields = all(field in ml_data[service]
                           for field in config['required_fields'])
            validation_status[service] = 'present' if has_fields else 'incomplete'

    return validation_status
```

#### Integration Point in main():
```python
# After Step 1 (load_unified_analysis)
service_status = validate_ml_services(unified_dict)

# Print validation report
print("\n🔍 ML SERVICE VALIDATION")
missing = [s for s, status in service_status.items() if status != 'present']
if missing:
    print(f"⚠️  Warning: {len(missing)} services missing/incomplete: {missing}")
    print("   Test may not reflect production behavior!")
```

### Phase 2: Complete Output Comparison

#### Replace feature validation with deep JSON comparison:

```python
def compare_complete_output(test_result: Dict[str, Any], prod_result: Dict[str, Any]) -> bool:
    """
    True mirror validation: Compare ENTIRE output with production.
    Don't validate features - just verify outputs are identical.
    """
    import json
    from deepdiff import DeepDiff

    # Compare the entire JSON structure
    diff = DeepDiff(prod_result, test_result, ignore_order=True,
                    significant_digits=6)  # Allow small float differences

    if not diff:
        print("✅ PERFECT MIRROR: Test output identical to production!")
        return True
    else:
        print("❌ DIFFERENCES FOUND:")

        # Report specific differences
        if 'values_changed' in diff:
            print("\n📊 Value differences:")
            for path, change in diff['values_changed'].items():
                print(f"  {path}:")
                print(f"    Production: {change['old_value']}")
                print(f"    Test:       {change['new_value']}")

        if 'dictionary_item_added' in diff:
            print(f"\n➕ Extra fields in test: {diff['dictionary_item_added']}")

        if 'dictionary_item_removed' in diff:
            print(f"\n➖ Missing fields in test: {diff['dictionary_item_removed']}")

        return False

def simplified_validation(test_result: Dict[str, Any], prod_path: Path) -> None:
    """
    Simple, complete validation - just compare outputs.
    This replaces all feature validation logic.
    """
    # Load production output
    with open(prod_path) as f:
        prod_result = json.load(f)

    # Compare complete outputs
    is_identical = compare_complete_output(test_result, prod_result)

    # Save comparison report
    comparison_report = {
        'is_identical': is_identical,
        'test_windows_count': len(test_result.get('temporal_windows', {}).get('middle_segments', [])),
        'prod_windows_count': len(prod_result.get('temporal_windows', {}).get('middle_segments', [])),
        'test_has_metadata': 'metadata' in test_result,
        'prod_has_metadata': 'metadata' in prod_result
    }

    output_path = Path(f'test_outputs/{video_id}_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\n💾 Comparison report saved to: {output_path}")
```

### Phase 3: Production Behavior Observation

#### Observe how production actually handles missing services:

```python
def observe_production_robustness(video_id: str):
    """
    Observe how production ACTUALLY handled any missing services.
    Maintains True Mirror - only analyzes what happened, doesn't create synthetic scenarios.
    """
    unified_dict = load_unified_analysis(video_id)

    # Check which services were actually missing in production
    EXPECTED_SERVICES = ['yolo', 'whisper', 'mediapipe', 'ocr',
                         'scene_detection', 'emotion_detection',
                         'audio_energy', 'deepface_gender']

    missing_in_prod = [s for s in EXPECTED_SERVICES
                       if s not in unified_dict['ml_data']
                       or not unified_dict['ml_data'][s]]

    if missing_in_prod:
        print(f"⚠️ Production ran with missing services: {missing_in_prod}")

        # Verify temporal_compute handled it correctly
        result = compute_temporal_windows(unified_dict)

        # Document how production handled the missing data
        print(f"✓ Temporal compute completed despite missing: {missing_in_prod}")

        # Log which features were affected
        if 'emotion_detection' in missing_in_prod:
            hook = result['temporal_windows']['hook']
            emotion_features = [f'{e}_ratio' for e in
                              ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']]
            print(f"  - Emotion features handled: {all(f in hook for f in emotion_features)}")
    else:
        print("✓ All ML services present in production run")
```

### Phase 4: Test Output Structure

#### Simplified test output directory:
```
test_outputs/
├── {video_id}_temporal_windows_test.json      # Main test output
├── {video_id}_ml_service_validation.json      # Service validation report
└── {video_id}_comparison_report.json          # Deep diff comparison with production
```

## Implementation Priority

1. **High Priority (Immediate)**
   - Add ML service validation to existing test
   - Replace feature validation with complete JSON comparison
   - Compare ALL outputs (not just hook window)

2. **Medium Priority (Next Sprint)**
   - Observe and document production's actual behavior with missing services
   - Create comprehensive comparison reports
   - Add data quality metrics

3. **Low Priority (Future)**
   - Automated regression testing
   - Performance benchmarking
   - Cross-video validation patterns

## Success Metrics

1. **Mirror Accuracy**: Test output exactly matches production output (100% equality)
2. **Service Awareness**: Test knows which ML services were present in production
3. **Difference Detection**: Test identifies any discrepancies with production
4. **Clarity**: Test output clearly shows what matched/differed and why

## Integration with Existing Test

The enhanced validation should be added to `test_temporal_compute_v2.py` by:

1. Insert ML service validation between Step 1 and Step 2
2. Replace current `validate_temporal_windows()` with `compare_complete_output()`
3. Remove feature validation entirely - just compare outputs
4. Add production behavior observation as optional Step 6

This maintains the "true mirror" philosophy by focusing on output equality rather than feature validation. The test becomes simpler and more accurate: same input → same function → must produce same output.