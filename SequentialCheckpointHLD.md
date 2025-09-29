# Sequential Checkpoint Contract - High Level Design (HLD)

**Version**: 1.0
**Last Updated**: January 2025
**Status**: Design Phase
**Scope**: Checkpoint & Resume Capability for RumiAI Sequential Processing

## 1. Executive Summary

### 1.1 Purpose
The Sequential Checkpoint Contract ensures reliable, resumable processing of large video batches (60-300 videos) by maintaining persistent progress state. This prevents complete reprocessing after failures and enables cost-effective ML pipeline operations.

### 1.2 Problem Statement
Currently, the RumiAI pipeline has no checkpoint mechanism, creating critical operational risks:
- **Complete Loss on Failure**: If processing fails at video 295 of 300, all 295 completed videos are lost
- **Wasted Compute Costs**: Reprocessing already-completed videos wastes GPU/CPU hours
- **No Failure Recovery**: System crashes, network issues, or quota limits require starting from scratch
- **Unpredictable Time-to-Completion**: Cannot estimate remaining work after interruption
- **ML Training Delays**: A single failure can delay training data availability by hours/days

### 1.3 Solution Overview
Implement a lightweight checkpoint system that tracks video processing progress, enabling seamless resume after any interruption.

---

## 2. Why Checkpointing is Critical for Sequential Processing

### 2.1 The Sequential Processing Reality
```
Video 1 → Video 2 → Video 3 → ... → Video 299 → [FAILURE] → Video 300
                                                    ↓
                                        Without checkpoints: Start over at Video 1
                                        With checkpoints: Resume at Video 300
```

### 2.2 Cost Impact Analysis

#### Without Checkpoints
- **Scenario**: Processing 300 videos, each takes ~2 minutes
- **Failure at video 280**: 560 minutes (9.3 hours) of processing lost
- **GPU costs**: ~$50-100 wasted per failure
- **Time cost**: Additional 9.3 hours before ML training can begin
- **Cascading impact**: Delays model improvements, feature releases

#### With Checkpoints
- **Same failure scenario**: 0 minutes lost
- **Cost savings**: $50-100 per failure event
- **Time savings**: Immediate resume, no reprocessing
- **Business impact**: Predictable ML training schedules

### 2.3 Failure Scenarios That Require Checkpoints

| Failure Type | Frequency | Impact Without Checkpoints |
|--------------|-----------|---------------------------|
| API Rate Limits | Daily | Lose entire day's progress |
| Memory Overflow | Per 50-100 videos | Reprocess all videos |
| Network Timeout | Random | Complete restart required |
| Service Crash | Weekly | Hours of work lost |
| Quota Exceeded | Monthly | Cannot complete batch |
| Power/System Failure | Rare but critical | Total data loss |

---

## 3. Business Value

### 3.1 Direct Benefits
- **Cost Reduction**: 80-90% reduction in reprocessing costs
- **Time Efficiency**: Complete 300-video batches reliably within SLA
- **Operational Confidence**: Start large batches without fear of total loss
- **Resource Optimization**: Free up compute for new work vs. reprocessing

### 3.2 Indirect Benefits
- **ML Model Quality**: More consistent training data availability
- **Developer Productivity**: Less time babysitting long-running processes
- **System Observability**: Clear progress tracking and failure patterns
- **Scalability Foundation**: Essential for moving from 60 to 300+ video batches

### 3.3 Risk Mitigation
Without checkpoints, the risk equation is:
```
Risk = (Probability of Failure) × (Cost of Complete Reprocessing)
     = 0.3 × $100 × 10 hours = Significant operational risk
```

With checkpoints:
```
Risk = (Probability of Failure) × (Cost of Resume from Checkpoint)
     = 0.3 × $0 × 0 hours = Negligible risk
```

---

## 4. Critical Requirements

### 4.1 What Makes Checkpointing Essential

#### 4.1.1 Progress Persistence
- **Requirement**: Progress must survive process termination
- **Why Critical**: Failures are unpredictable and often terminal
- **Without This**: Every failure means complete data loss

#### 4.1.2 Granular State Tracking
- **Requirement**: Track individual video completion status
- **Why Critical**: Videos have different processing costs (30s vs 10min)
- **Without This**: Cannot optimize resume strategy

#### 4.1.3 Failure Isolation
- **Requirement**: One video's failure shouldn't affect others
- **Why Critical**: Some videos may have corrupt data
- **Without This**: Single bad video blocks entire pipeline

#### 4.1.4 Idempotent Processing
- **Requirement**: Safe to retry any video without side effects
- **Why Critical**: Enables confident resume operations
- **Without This**: Risk of duplicate data or corruption

### 4.2 Operational Requirements

#### 4.2.1 Zero-Configuration Resume
- **Need**: Automatic detection and resume from last checkpoint
- **Impact**: Reduces operator intervention and human error

#### 4.2.2 Progress Visibility
- **Need**: Real-time progress monitoring
- **Impact**: Enables capacity planning and deadline management

#### 4.2.3 Failure Threshold Management
- **Need**: Stop processing if too many failures occur
- **Impact**: Prevents wasted compute on systematically failing batches

---

## 5. Use Cases Requiring Checkpoints

### 5.1 Daily ML Training Pipeline
```
6:00 AM: Start processing 300 videos for hashtag #BookTok
10:00 AM: API rate limit hit at video 120
10:01 AM: [With checkpoints] Resume automatically when limit resets
         [Without checkpoints] Lose 4 hours of work, restart at 6PM
2:00 PM: All 300 videos complete
3:00 PM: ML training begins on schedule
```

### 5.2 Emergency Interruption
```
Scenario: Emergency system maintenance required
Current: Processing video 250 of 300
Action: Graceful shutdown
Result with checkpoints: Resume at video 250 after maintenance
Result without: Lose 8+ hours of processing
```

### 5.3 Cascading Service Failures
```
Video 50: OCR service timeout - marked failed, continue
Video 51-100: Process normally
Video 101: FEAT service OOM - marked failed, continue
Video 102-299: Process normally
Video 300: Complete

With checkpoints: 298 successful videos ready for ML
Without checkpoints: Would have stopped at first failure
```

---

## 6. Architecture Placement: Main Production Code

### 6.1 Where Checkpoint Lives
The Sequential Checkpoint system belongs in the **main production pipeline** (e.g., rumiai_runner.py), NOT in the ML development layer:

```
ML Development Layer (MLROADMAP.md):
├── Duration bucketing algorithms
├── Model training pipelines
└── Feature engineering
         ↑
    [Consumes processed data]
         ↑
Main Production Pipeline (rumiai_runner.py):
├── Video processing orchestration
├── ML service execution (YOLO, Whisper, etc.)
├── **CHECKPOINT SYSTEM** ← Lives here
└── Output storage
```

### 6.2 Why It's Production Code, Not ML Code
1. **Wraps core processing loop**: Checkpoints track video processing completion
2. **ML-agnostic**: Doesn't know about buckets, models, or training
3. **Infrastructure concern**: Like logging or monitoring - enables reliability
4. **ML layer just consumes**: ML assumes videos are processed, checkpoint ensures they are

### 6.3 Integration Point
```python
# In rumiai_runner.py (production code)
checkpoint = SequentialCheckpoint(hashtag)
for video in video_list:
    checkpoint.start_video(video)
    try:
        process_all_services(video)  # YOLO, Whisper, etc.
        checkpoint.complete_video(video)
    except Exception as e:
        checkpoint.fail_video(video, str(e))
```

The ML layer never touches checkpoints - it just reads the completed outputs.

## 7. Why This is Different from Batch Controllers

### 7.1 Sequential vs Parallel
- **Batch Controller**: Manages parallel execution, work distribution
- **Checkpoint System**: Tracks sequential progress, enables resume
- **Key Difference**: We're not orchestrating, we're persisting

### 6.2 Complexity Comparison
| Aspect | Batch Controller | Checkpoint System |
|--------|-----------------|-------------------|
| Lines of Code | 500-1000 | 100-200 |
| External Dependencies | Message queues, coordination | Just filesystem |
| Failure Modes | Complex distributed failures | Simple file I/O |
| Implementation Time | Weeks | Hours |
| Maintenance Burden | High | Minimal |

### 6.3 Why Checkpoints are Sufficient
For sequential processing, you need:
- ✅ Know where you left off (checkpoint provides)
- ✅ Skip completed work (checkpoint provides)
- ✅ Track failures (checkpoint provides)
- ❌ Coordinate parallel workers (not needed)
- ❌ Distribute load (not needed)
- ❌ Manage queues (not needed)

---

## 7. Impact of NOT Having Checkpoints

### 7.1 Immediate Consequences
- **Developer Anxiety**: Fear of starting large batches
- **Manual Workarounds**: Breaking 300 videos into 10x30 manual runs
- **Weekend Work**: Babysitting long-running processes
- **Delayed Launches**: ML features delayed by unreliable data pipeline

### 7.2 Long-term Consequences
- **Technical Debt**: Ad-hoc recovery scripts accumulate
- **Operational Fragility**: System becomes increasingly unreliable
- **Scaling Limitations**: Cannot move from 300 to 1000+ videos
- **Competitive Disadvantage**: Slower iteration on ML models

### 7.3 Hidden Costs
- **Opportunity Cost**: Engineers managing failures instead of building features
- **Morale Impact**: Frustration from repeatedly losing work
- **Customer Impact**: Delayed ML improvements affect product quality

---

## 8. Success Criteria

### 8.1 Immediate Success (Day 1)
- ✅ Can resume from any failure point
- ✅ No videos processed twice
- ✅ Clear progress visibility

### 8.2 Operational Success (Week 1)
- ✅ 300-video batches complete reliably
- ✅ 90% reduction in reprocessing
- ✅ Unattended overnight runs succeed

### 8.3 Business Success (Month 1)
- ✅ ML training data always available on schedule
- ✅ Confident scaling to larger batches
- ✅ Measurable compute cost savings

---

## 9. Why This Should Be the #1 Priority

### 9.1 Highest Impact-to-Effort Ratio
- **Implementation**: ~4 hours
- **Impact**: Saves days of reprocessing per month
- **ROI**: 100x return on implementation time

### 9.2 Enables Everything Else
Without reliable checkpoint/resume:
- Output validation is less valuable (still lose progress on failures)
- Resource limits don't help (still can't resume)
- ML training remains unpredictable

### 9.3 Immediate Pain Relief
This solves the most acute current pain:
> "What if it fails at video 295?"

With checkpoints, the answer changes from "start over" to "no problem."

---

## 10. Conclusion

The Sequential Checkpoint Contract is not a nice-to-have optimization—it's a fundamental requirement for operating a reliable ML pipeline at scale. Without it, the RumiAI pipeline remains fragile, expensive, and unpredictable. With it, sequential processing of hundreds of videos becomes a solved problem, enabling focus on ML model improvements rather than operational firefighting.

**The question isn't whether to implement checkpointing, but why it hasn't been implemented already.**

---

## Document History
- v1.0 (2025-01-26): Initial HLD created focusing on business value and operational necessity