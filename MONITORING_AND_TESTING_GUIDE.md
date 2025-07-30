# RumiAI v2 Monitoring and Testing Guide

## Table of Contents
1. [Memory Monitoring](#memory-monitoring)
2. [Cost Tracking](#cost-tracking)
3. [Testing Strategy](#testing-strategy)
4. [Feature Flags](#feature-flags)
5. [Performance Metrics](#performance-metrics)
6. [Troubleshooting](#troubleshooting)

## Memory Monitoring

### Overview
Memory monitoring is crucial for processing 2-minute videos without running out of RAM or causing system instability. The system tracks memory usage at multiple points and takes corrective action when thresholds are exceeded.

### Implementation Details

#### 1. Memory Usage Collection
**File**: `/scripts/rumiai_runner.py`

```python
def _get_memory_usage(self) -> Dict[str, float]:
    """Get current memory usage statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()
    
    return {
        'process_rss_gb': memory_info.rss / 1024**3,  # Resident Set Size (actual RAM used)
        'process_vms_gb': memory_info.vms / 1024**3,  # Virtual Memory Size
        'system_percent': virtual_memory.percent,     # System-wide memory usage %
        'system_available_gb': virtual_memory.available / 1024**3
    }
```

**Metrics Explained**:
- **RSS (Resident Set Size)**: Actual physical RAM currently used by the process
- **VMS (Virtual Memory Size)**: Total virtual memory allocated (includes swapped memory)
- **System Percent**: Percentage of total system memory in use
- **Available Memory**: How much RAM is available system-wide

#### 2. Memory Threshold Checking
```python
def _check_memory_threshold(self, threshold_gb: float = 4.0) -> bool:
    """Check if we're approaching memory limits."""
    memory = self._get_memory_usage()
    
    if memory['process_rss_gb'] > threshold_gb:
        logger.warning(f"High memory usage: {memory['process_rss_gb']:.1f}GB")
        gc.collect()  # Force garbage collection
        return True
    
    if memory['system_percent'] > 90:
        logger.warning(f"System memory critically low: {memory['system_percent']:.1f}%")
        return True
    
    return False
```

**Thresholds**:
- **Process threshold**: 4GB default (configurable)
- **System threshold**: 90% of total RAM
- **Action**: Forces Python garbage collection when exceeded

#### 3. Memory Monitoring Points

Memory is monitored at these critical points:

1. **Before/After ML Analysis**:
   ```python
   # Before ML analysis
   initial_memory = self._get_memory_usage()
   logger.info(f"Memory before ML: {initial_memory['process_rss_gb']:.1f}GB")
   
   ml_results = await self._run_ml_analysis(video_id, video_path)
   
   # After ML analysis
   post_ml_memory = self._get_memory_usage()
   logger.info(f"Memory after ML: {post_ml_memory['process_rss_gb']:.1f}GB")
   ```

2. **Between Claude Prompts**:
   ```python
   # Check memory after each prompt
   if self._check_memory_threshold(threshold_gb=3.5):
       print("⚠️ Memory threshold reached, cleaning up...")
       gc.collect()
       await asyncio.sleep(2)  # Give system time to free memory
   ```

3. **Final Report**:
   ```python
   'memory_usage': {
       'final_process_gb': final_memory['process_rss_gb'],
       'peak_process_gb': max(final_memory['process_rss_gb'], 4.0),
       'system_percent': final_memory['system_percent']
   }
   ```

### Memory Management Strategies

1. **Garbage Collection**: Forced GC when thresholds exceeded
2. **Processing Delays**: 2-second pause after GC to allow memory recovery
3. **Early Warning**: Logs warnings before critical levels
4. **Metrics Tracking**: Records memory usage for analysis

## Cost Tracking

### Overview
Cost tracking monitors API usage expenses, primarily for Claude API calls. The system tracks costs at multiple levels: per-prompt, per-video, and aggregate.

### Cost Calculation

#### 1. Model Pricing
**File**: `/rumiai_v2/api/claude_client.py`

```python
MODEL_PRICING = {
    "claude-3-haiku-20240307": {
        "input": 0.25,    # $ per million tokens
        "output": 1.25    # $ per million tokens
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,    # $ per million tokens
        "output": 15.00   # $ per million tokens
    }
}
```

#### 2. Per-Request Cost Calculation
```python
# Calculate cost based on actual token usage
input_cost = (input_tokens / 1_000_000) * input_price
output_cost = (output_tokens / 1_000_000) * output_price
total_cost = input_cost + output_cost

# Example for Sonnet with 1000 input + 500 output tokens:
# Input: (1000 / 1,000,000) * $3.00 = $0.003
# Output: (500 / 1,000,000) * $15.00 = $0.0075
# Total: $0.0105
```

#### 3. Cost Tracking Levels

**Per-Prompt Tracking**:
```python
self.video_metrics.record_prompt_cost(prompt_type.value, result.estimated_cost)

# Stored in prompt_details:
"creative_density": {
    "success": true,
    "tokens": 5692,
    "cost": 0.025524,  # $0.026
    "time": 9.97
}
```

**Per-Video Summary**:
```python
# In final report
"total_cost": 0.057195,  # Sum of all prompt costs
"total_tokens": 39515,   # Sum of all tokens used
```

**Aggregate Metrics**:
```python
# VideoProcessingMetrics tracks across all videos
'prompt_costs': {
    'creative_density': 0.025524,
    'emotional_journey': 0.021156,
    # ... etc
},
'total_cost': 0.182745,
'cost_per_video': 0.0261  # Average
```

### Cost Optimization Features

1. **Dynamic Model Selection**:
   - Use cheaper Haiku model by default
   - Switch to Sonnet only when needed via feature flag

2. **Prompt Size Monitoring**:
   ```python
   print(f"📏 Payload size: {size_kb}KB")
   ```

3. **Early Failure Detection**:
   - Stops processing if prompts fail
   - Avoids wasting API calls

4. **Cost Reporting**:
   ```python
   print(f"💰 Total cost: ${batch.get_total_cost():.4f}")
   ```

## Testing Strategy

### Feature Flag System

**File**: `/rumiai_v2/config/settings.py`

```python
# ML Enhancement Feature Flags
self.use_ml_precompute = os.getenv('USE_ML_PRECOMPUTE', 'false').lower() == 'true'
self.use_claude_sonnet = os.getenv('USE_CLAUDE_SONNET', 'false').lower() == 'true'
self.output_format_version = os.getenv('OUTPUT_FORMAT_VERSION', 'v1')
```

### Testing Modes

#### 1. Legacy Mode (Baseline)
```bash
# All flags disabled - original behavior
./venv/bin/python scripts/rumiai_runner.py <video_url>
```

**Characteristics**:
- Uses MLDataExtractor for basic metrics
- Haiku model ($0.25/$1.25 per million tokens)
- Legacy output format
- ~5KB prompt sizes

#### 2. Precompute Only
```bash
export USE_ML_PRECOMPUTE=true
./venv/bin/python scripts/rumiai_runner.py <video_url>
```

**Characteristics**:
- Uses precompute functions (245+ metrics)
- Still uses Haiku model
- Legacy output format
- ~13KB prompt sizes

#### 3. Sonnet Model Test
```bash
export USE_CLAUDE_SONNET=true
./venv/bin/python scripts/rumiai_runner.py <video_url>
```

**Characteristics**:
- Uses Sonnet 3.5 model ($3/$15 per million tokens)
- 12x more expensive but more capable
- Better for complex analysis

#### 4. 6-Block Format Test
```bash
export OUTPUT_FORMAT_VERSION=v2
./venv/bin/python scripts/rumiai_runner.py <video_url>
```

**Characteristics**:
- Outputs structured 6-block JSON
- Requires response validation
- Better for ML training

#### 5. Full Enhancement Mode
```bash
export USE_ML_PRECOMPUTE=true
export USE_CLAUDE_SONNET=true
export OUTPUT_FORMAT_VERSION=v2
./venv/bin/python scripts/rumiai_runner.py <video_url>
```

### Response Validation

**File**: `/rumiai_v2/validators/response_validator.py`

The validator ensures 6-block responses are properly structured:

1. **JSON Parsing**: Validates response is valid JSON
2. **Block Presence**: Checks all 6 blocks exist
3. **Block Structure**: Ensures each block is a dictionary
4. **Field Validation**: Skipped for prompt-specific formats
5. **Quality Checks**: 
   - Minimum content length (500 chars)
   - No empty blocks

### Integration Test Checklist

**File**: `/RUMIAI_RUNNER_UPGRADE_CHECKLIST.md`

#### Phase 1: Unit Testing
```bash
# Test individual components
python -m pytest tests/test_precompute_functions.py
python -m pytest tests/test_output_adapter.py
python -m pytest tests/test_prompt_manager.py
```

#### Phase 2: Integration Testing
```bash
# Test with different video types
./test_integration.sh --video-type short  # <30s
./test_integration.sh --video-type medium # 30-60s
./test_integration.sh --video-type long   # 60-120s
```

#### Phase 3: Load Testing
```bash
# Test memory usage with long videos
./test_memory_usage.sh --duration 120 --monitor
```

#### Phase 4: Cost Analysis
```bash
# Compare costs between modes
./compare_costs.sh --legacy vs --enhanced
```

## Performance Metrics

### Metrics Collection

**File**: `/rumiai_v2/utils/metrics.py`

The system tracks:

1. **Timing Metrics**:
   - Total processing time
   - ML analysis time
   - Per-prompt execution time
   - Download/scraping time

2. **Success Metrics**:
   - Videos processed successfully
   - Prompt success rate
   - ML service availability

3. **Resource Metrics**:
   - Memory usage (RSS, VMS)
   - CPU percentage
   - GPU availability

### GPU Monitoring

```python
def _verify_gpu(self) -> None:
    """Verify GPU/CUDA availability at startup."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU available: {device_name} with {memory:.1f}GB VRAM")
            print(f"🎮 GPU: {device_name} ({memory:.1f}GB VRAM)")
```

**GPU Benefits**:
- 10-30x speedup for ML models
- Essential for 2-minute video processing
- Reduces memory pressure via batch processing

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors
**Symptoms**: Process killed, system freeze
**Solutions**:
- Reduce `threshold_gb` in `_check_memory_threshold`
- Increase prompt delays
- Process shorter videos
- Add more RAM or use GPU

#### 2. High API Costs
**Symptoms**: Unexpected bills
**Solutions**:
- Use Haiku model for testing
- Monitor payload sizes
- Implement cost limits
- Cache results locally

#### 3. Validation Failures
**Symptoms**: "Invalid response format" errors
**Solutions**:
- Check prompt templates for JSON instructions
- Verify model is returning pure JSON
- Update validator for new block names
- Check for markdown in responses

#### 4. Slow Processing
**Symptoms**: Timeouts, long waits
**Solutions**:
- Verify GPU is detected
- Check network speed
- Reduce prompt complexity
- Use dynamic timeouts

### Monitoring Commands

```bash
# Watch memory usage
watch -n 1 'ps aux | grep rumiai_runner'

# Monitor GPU usage
nvidia-smi -l 1

# Check costs
grep "Total cost" logs/rumiai_*.log | awk '{sum += $4} END {print sum}'

# Validate outputs
jq . insights/*/creative_density/*_complete_*.json
```

### Debug Environment Variables

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG

# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Dry run mode (no API calls)
export DRY_RUN=true

# Cost limit per video
export MAX_COST_PER_VIDEO=0.10
```

## Best Practices

1. **Start with Legacy Mode**: Establish baseline
2. **Enable Features Gradually**: One flag at a time
3. **Monitor Resources**: Watch memory and costs
4. **Test Various Videos**: Different lengths and complexity
5. **Document Results**: Track improvements and issues
6. **Set Limits**: Configure thresholds appropriately
7. **Regular Cleanup**: Clear temp files and old results

## Conclusion

The monitoring and testing system provides comprehensive visibility into:
- Memory usage and management
- API costs and optimization
- Processing performance
- System health

Use feature flags to safely test enhancements while maintaining production stability.