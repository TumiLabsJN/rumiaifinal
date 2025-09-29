# Resource Limits Contract - High Level Design

**Version**: 1.0
**Last Updated**: January 2025
**Status**: Design Phase
**Parent Document**: [ServiceContractsv2.md](./ServiceContractsv2.md)

## Executive Summary

The Resource Limits Contract enforces timeouts and memory limits on ML services to prevent runaway processes, memory exhaustion, and pipeline stalls. This contract is critical for maintaining pipeline stability when processing videos sequentially, especially for resource-intensive services like FEAT (which currently consumes 43% of pipeline time).

## Why Timeout Service Contracts Are Critical

### The Sequential Processing Risk
In our ML pipeline processing TikTok videos, services run sequentially. This creates a cascade risk where a single hung service can destroy the entire batch operation:

1. **The FEAT Problem**: FEAT already consumes 43% of our processing time (empirically measured). Without timeouts, if FEAT hangs on video #5 of 60, we lose the entire batch - no results, wasted compute, angry users.

2. **The Multiplication Effect**:
   - Normal: 60 videos × 3 min = 3 hours
   - One hung service: ∞ (pipeline never completes)
   - Business impact: $0 revenue from that batch

3. **Resource Protection**: Hung processes don't just waste time - they hold memory, GPU resources, and file handles. In cloud environments, this directly translates to:
   - Wasted compute costs (paying for stuck instances)
   - Blocked GPU resources ($$$)
   - Memory leaks accumulating over time

4. **User Experience**: A user uploads their video expecting results in ~3 minutes. Without timeouts:
   - They wait indefinitely
   - Support tickets pile up
   - Trust erodes
   - They switch to competitors

5. **Operational Reality**: In production, things go wrong:
   - Corrupted video frames cause infinite loops
   - Network calls to external APIs hang
   - Memory leaks cause gradual degradation
   - Edge cases trigger unexpected behavior

### The Business Case
Without timeout contracts, one bad video or one edge case bug can:
- Block 59 other paying customers
- Waste hours of expensive GPU time
- Generate support costs exceeding revenue
- Damage brand reputation

With timeout contracts:
- Worst case: One video fails, 59 succeed (98.3% success)
- Resources freed for next batch
- Clear error messages for debugging
- Predictable SLAs for customers

## Problem Statement

### Current Issues
1. **No Timeouts**: Services can run indefinitely, blocking the pipeline
2. **No Memory Limits**: Services can exhaust system memory causing crashes
3. **No Resource Monitoring**: No visibility into service resource consumption
4. **FEAT Bottleneck**: Single service consuming 43% of total processing time with no protection

### Impact
- Single hung service can block entire video batch processing
- Memory leaks accumulate over 60+ video batches
- No early warning system for performance degradation
- Silent failures when system resources exhausted

## Solution Overview

### Core Components

#### 1. Service-Specific Limits
Each ML service gets tailored resource limits based on empirical measurements. These are **fixed timeouts** that do not scale with video duration, providing predictable behavior regardless of input length. Memory limits are set **conservatively above measured usage** to accommodate edge cases and future growth:

| Service | Timeout | Memory Limit | Warning Threshold |
|---------|---------|--------------|-------------------|
| FEAT | 5 min | 3.0 GB | 2.5 min |
| Whisper | 3 min | 2.0 GB | 1.5 min |
| OCR | 2 min | 0.5 GB | 1 min |
| YOLO | 1 min | 1.5 GB | 30s |
| MediaPipe | 2 min | 0.5 GB | 1 min |
| DeepFace | 1 min | 1.0 GB | 30s |
| Audio Energy | 1 min | 0.5 GB | 30s |
| Scene Detection | 1 min | 0.5 GB | 30s |

#### 2. Monitoring System
- Real-time resource tracking during service execution
- Warning alerts at exactly 50% of timeout threshold (single-tier, log-only)
- Memory usage snapshots every 5 seconds (fixed interval for all services)
- Performance metrics collection for optimization

#### 3. Enforcement Mechanisms
- Hard timeout with process termination
- Memory limit checks with immediate termination
- Resource cleanup after service completion
- Pipeline termination on any service failure (all services critical)

## Design Decisions

### Decision 1: Fixed vs Dynamic Timeouts
**Decision**: Use fixed timeouts that do not scale with video duration.

**Rationale**:
- **Predictability**: Fixed timeouts provide consistent behavior and clear SLAs
- **Simplicity**: No complex calculations based on video metadata
- **Protection**: Prevents edge cases where long videos could extend timeouts indefinitely
- **Testing**: Easier to test and validate with known boundaries

**Implementation**: Each service has a predetermined maximum execution time regardless of whether processing a 15-second or 60-second video. Services that naturally scale with video length (like Whisper for transcription) have their fixed timeout set to accommodate the longest expected video.

### Decision 2: Conservative Memory Limits
**Decision**: Use conservative memory limits well above measured usage for future-proofing.

**Rationale**:
- **Model Evolution**: ML models may grow larger over time (e.g., upgraded YOLO versions)
- **Edge Cases**: Unusual videos may require more memory than typical test videos
- **Safety Margin**: Prevents false positives from transient memory spikes
- **Future Features**: New processing features may increase memory requirements

**Measured vs Allocated**:
- Emotion Detection: 1.1GB measured → 3.0GB allocated (3x buffer)
- YOLO: 960MB measured → 1.5GB allocated (1.5x buffer)
- MediaPipe: 400MB measured → 0.5GB allocated (1.25x buffer)

This approach prioritizes stability over resource optimization, appropriate for a production system where reliability is paramount.

### Decision 3: Single-Threshold Warning Strategy
**Decision**: Implement a single warning at 50% of timeout with log-only behavior.

**Rationale**:
- **Simplicity**: One threshold is easy to understand and configure
- **Early Detection**: 50% provides adequate time to observe slowdowns
- **Low Overhead**: Simple logging doesn't impact performance
- **Avoid Complexity**: Multi-tier systems add complexity without proven benefit

**Implementation**:
- Warning triggers at exactly 50% of service timeout
- Logs as WARNING level with service name and elapsed time
- No automated actions taken (passive monitoring)
- Example: FEAT warns at 2.5 min, Whisper at 1.5 min

**Future Enhancement Path**: Can add multi-tier or adaptive thresholds after collecting baseline performance data.

### Decision 4: Fixed 5-Second Monitoring Interval
**Decision**: Use a uniform 5-second interval for all resource monitoring checks.

**Rationale**:
- **Low Overhead Priority**: Minimizes CPU usage (~0.1%) over faster detection
- **Sufficient Granularity**: 5 seconds is adequate for services with 1-5 minute timeouts
- **Simplicity**: Single interval for all services reduces complexity
- **Proven Adequate**: Current ThreadMonitor in video_analyzer.py uses similar interval successfully

**Trade-offs Accepted**:
- May miss memory spikes lasting <5 seconds (acceptable risk)
- 5-second maximum detection delay (acceptable for our timeout ranges)
- Uniform approach may be suboptimal for some services (simplicity wins)

**Implementation**: Background async task wakes every 5 seconds to check memory, CPU time, and elapsed time for all active services.

### Decision 5: All-Critical Fail-Fast Strategy
**Decision**: All ML services are critical - any resource limit failure terminates the entire video analysis.

**Rationale**:
- **Data Integrity**: RumiAI's 60+ features require complete analysis for reliable insights
- **Business Value**: Partial results have no business value - customers need full feature sets
- **Predictable Behavior**: Clear success/failure states, no ambiguous partial results
- **Simplified Error Handling**: No complex logic for determining which services can fail

**Implementation**:
- Any service timeout or memory limit breach = immediate pipeline termination
- Return clear error message identifying which service and limit was exceeded
- No retry logic - fix the underlying issue (video complexity, resource allocation)
- Video marked as "failed" in batch processing, continue with next video

**Error Response Format**:
```json
{
  "status": "failed",
  "error": "Resource limit exceeded",
  "failed_service": "feat",
  "limit_type": "timeout",
  "limit_value": "300s",
  "actual_value": "300s"
}
```

## Architecture Design

### Resource Contract Flow

```
┌─────────────────────────────────────────────┐
│           Resource Contract Manager          │
├─────────────────────────────────────────────┤
│                                              │
│  1. Service Request                          │
│     ↓                                        │
│  2. Load Service Limits                      │
│     ↓                                        │
│  3. Start Resource Monitor (Background)      │
│     ↓                                        │
│  4. Execute Service (with timeout wrapper)   │
│     ↓                                        │
│  ┌──────────────┐    ┌──────────────┐       │
│  │   Success    │    │   Failure    │       │
│  └──────────────┘    └──────────────┘       │
│         ↓                    ↓               │
│  5. Stop Monitor      5. Kill Process        │
│     ↓                    ↓                   │
│  6. Log Metrics      6. Log Failure          │
│     ↓                    ↓                   │
│  7. Return Result    7. TERMINATE PIPELINE   │
│                                              │
└─────────────────────────────────────────────┘
```

### Monitoring Architecture

```
Resource Monitor (Async Task)
├── Check Every 5 Seconds (Fixed Interval)
│   ├── Memory Usage
│   │   └── Compare to Limit
│   ├── Execution Time
│   │   └── Check Warning Threshold (50%)
│   └── Process Health
│       └── Verify Still Running
├── Alert on Threshold
│   ├── Log Warning (No Action)
│   └── Continue Monitoring
└── Enforce Limits
    ├── Memory Exceeded → Kill Process
    └── Timeout Reached → Terminate
```

## Contract Placement in Pipeline

The Resource Limits Contract operates at the **ML Service level**:

```
Video → [ML Services ← CONTRACT HERE] → Timeline Builder → Temporal Compute → ML Training
```

This placement ensures protection at the source of resource consumption, before aggregation.

## Integration Points

### 1. Video Analyzer Integration
```python
# Before (no protection)
result = await service.run(video_path)

# After (with resource limits)
result = await ResourceContract.run_with_limits(
    service_name='feat',
    coroutine=service.run(video_path)
)
```

### 2. Batch Processor Integration
- Apply limits to each video in batch
- On failure: mark video as failed, continue with next video
- Track failure patterns across batch for system health monitoring

### 3. Error Handling Integration
- Terminate pipeline immediately on resource failures
- Provide detailed failure context (service, limit type, values)
- Enable circuit breaker decisions based on failure patterns

## Implementation Strategy

### Phase 1: Core Implementation (2 hours)
1. Create ResourceContract class
2. Implement timeout wrapper
3. Add memory monitoring
4. Basic logging

### Phase 2: Service Configuration (1 hour)
1. Configure fixed limits per service (non-scaling with video duration)
2. Add warning thresholds (50% of timeout value)
3. Set up metrics collection

### Phase 3: Integration (1 hour)
1. Integrate with video_analyzer.py
2. Add to batch processing
3. Connect error handling

### Phase 4: Testing (1 hour)
1. Test timeout enforcement
2. Test memory limits
3. Validate metrics collection
4. Edge case testing

## Success Criteria

### Functional Requirements
- [ ] All services enforce timeout limits
- [ ] Memory limits prevent system exhaustion
- [ ] Warning system provides early alerts
- [ ] Metrics collected for all services

### Performance Requirements
- [ ] <100ms overhead per service call
- [ ] <50MB memory for monitoring system
- [ ] Zero false positive terminations
- [ ] 100% timeout enforcement accuracy

### Operational Requirements
- [ ] Clear error messages on limit exceeded
- [ ] Detailed resource usage logs
- [ ] Fast failure with immediate termination
- [ ] Automatic resource cleanup

## Risk Mitigation

### Risk 1: False Positive Terminations
**Mitigation**: Set conservative limits initially, monitor actual usage, adjust based on data

### Risk 2: Resource Monitoring Overhead
**Mitigation**: Async monitoring, fixed 5-second intervals prioritizing low overhead, lightweight checks

### Risk 3: Service-Specific Edge Cases
**Mitigation**: Configurable limits per service, override capability for special cases

### Risk 4: Video Duration Variance
**Mitigation**: Fixed timeouts set conservatively to handle maximum expected video duration (60 seconds), preventing timeout scaling complexity while ensuring all videos can be processed

## Monitoring & Metrics

### Key Metrics to Track
1. **Service Duration Percentiles** (p50, p90, p99)
2. **Memory Usage Peaks** per service
3. **Timeout Incidents** per service
4. **Resource Warnings** frequency
5. **Performance Impact** (overhead measurement)

### Alert Thresholds
- **Timeout Warning**: Logged at 50% of limit (e.g., 2.5 min for 5 min timeout)
- **Memory Warning**: Logged at 90% of limit (close to exhaustion)
- **Repeated Failures**: Alert if same service times out 3+ times in 10 minutes
- **Pattern Detection**: Flag unusual consumption only after baseline established

## Future Enhancements

### Version 1.1
- Dynamic limit adjustment based on video characteristics
- Predictive resource allocation
- Resource reservation system

### Version 2.0
- Distributed resource management
- GPU resource tracking
- Network bandwidth limits
- Disk I/O monitoring

## Dependencies

### Required Components
- Python asyncio for timeout management
- psutil for resource monitoring
- logging for metrics collection

### Performance Baseline
- Based on ServicesPerformance.md measurements
- Actual memory usage: ~2.7GB total for full pipeline
- Conservative limits: ~9GB total allocation (3.3x safety factor)

### Related Contracts
- **Output Validation Contract**: Validates data after resource-limited execution
- **Circuit Breaker Contract**: Uses resource failures for circuit decisions
- **Input Validation Contract**: Provides video metadata for resource estimation

## Testing Strategy

### Unit Tests
- Timeout enforcement validation
- Memory limit checking
- Warning threshold alerts
- Metrics collection accuracy

### Integration Tests
- End-to-end service execution with limits
- Batch processing with resource constraints
- Failure recovery scenarios
- Resource cleanup validation

### Performance Tests
- Monitoring overhead measurement
- Concurrent service resource tracking
- Long-running batch simulations

## Documentation

### For Developers
- Integration guide with code examples
- Configuration reference
- Troubleshooting guide

### For Operations
- Resource limit tuning guide
- Monitoring dashboard setup
- Alert response procedures

## Approval & Sign-off

- [ ] Technical Design Review
- [ ] Security Review (resource access)
- [ ] Performance Impact Assessment
- [ ] Operations Readiness Review

---

## References

- [ServiceContractsv2.md](./ServiceContractsv2.md) - Parent design document
- [InstrumentationResults.md](../InstrumentationResults.md) - Performance measurements
- [SystemArchitecturev2.md](../documentation_migration/services/SystemArchitecturev2.md) - System architecture