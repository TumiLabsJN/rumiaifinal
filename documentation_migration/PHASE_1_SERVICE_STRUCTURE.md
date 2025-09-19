# PHASE 1: Service Documentation Structure

## ⚠️ CRITICAL REMINDER - LEGACY DANGER
**ALL legacy documentation contains outdated precompute references.**
This includes but is not limited to:
- FRAME_PROCESSING_PYTHON_ONLY.md (frame sampling claims)
- PersonFraming.md, GestureAnalysis.md, etc. (service descriptions)
- Any document referencing precompute functions

**VERIFY EVERYTHING through code inspection. Do NOT trust legacy claims.**

## 📋 Phase 1 Scope
Document HOW services work technically, NOT their ML value (Phase 2) or system architecture (Phase 3).

## 📁 Documents to Create
1. **VisionServices.md** - YOLO, MediaPipe, OCR, Scene Detection
2. **AudioServices.md** - Whisper, Audio Energy
3. **AnalysisServices.md** - FEAT, DeepFace
4. **MetadataServices.md** - Hashtag analysis, Engagement metrics

## 📝 DOCUMENT TEMPLATE

Use this exact structure for each service document:

```markdown
# [Service Group Name] (e.g., Vision Services)

## ⚠️ Legacy Information Warning
This document may reference legacy sources (FRAME_PROCESSING_PYTHON_ONLY.md, etc.).
ALL information from legacy sources has been verified through code inspection.
Claims marked [LEGACY] required special verification.

## 🔄 Parallel Execution Architecture
**IMPORTANT**: All services in this group run concurrently, NOT sequentially.
- Services are launched in parallel through their orchestrator
- Individual service timing cannot be isolated in production
- Services compete for system resources (CPU, memory, I/O)
- Performance measurements show aggregate time for all parallel services

## 📦 Batch Processing Clarification
**CRITICAL DISTINCTION** - Two types of "batching":
1. **Frame Batching** (WITHIN one video):
   - Processing multiple frames together for efficiency
   - Example: YOLO processes 10 frames per batch
   - Example: MediaPipe processes 20 frames per batch
   - This is an optimization technique for single video processing

2. **Video Batching** (NOT IMPLEMENTED):
   - RumiAI processes ONE video at a time
   - No parallel processing of multiple videos
   - Each video goes through the complete pipeline sequentially

## 📊 Service Overview Matrix
| Service | Purpose | Status | Currently Using | GPU Compatible | Output Type | Self-Contained |
|---------|---------|--------|-----------------|----------------|-------------|----------------|
| Service1 | What it does | ✅/⚠️/❌ | CPU/GPU | ✅/⚠️/❌ | Timeline/ML Data | ✅/❌ |
| Service2 | What it does | ✅/⚠️/❌ | CPU/GPU | ✅/⚠️/❌ | Timeline/ML Data | ✅/❌ |

---

# [Individual Service Name] (e.g., MediaPipe Service)

## 🔄 Execution Context
**This service runs in parallel with other [vision/audio/analysis] services through `[orchestrator_file].py`.**
- Launched concurrently with [list other parallel services]
- [Resource usage notes: CPU/GPU/memory competition]
- Cannot be timed individually in production without instrumentation

## 🎯 Service Purpose
- **Single sentence**: What this service does
- **Input type**: frames/audio/metadata
- **Output type**: Brief description of JSON structure

## ⚡ Performance Profile
```
Execution Time (To Be Confirmed - Pending Instrumentation Tests):
- 60-second video: TBC seconds
- 120-second video: TBC seconds

Resource Usage (To Be Confirmed - Pending Instrumentation Tests):
- Memory: TBC GB peak
- CPU: TBC% average (TBC cores)
- GPU Compatible: ✅ Yes (CUDA) / ⚠️ Optional / ❌ No (CPU only)
- GPU Usage (if compatible): TBC% average

Configuration:
- Parallelizable: Yes/No (within single video)
- Frame Batching: Yes/No (processes X frames per batch)
- Video Processing: Sequential (one video at a time)
- Current Status: ✅ Optimized / ⚠️ Needs Work / ❌ Critical Issues

IMPORTANT DISTINCTION:
- Frame Batching: Processing multiple frames together within ONE video (for efficiency)
- Video Processing: RumiAI ALWAYS processes one video at a time (never parallel videos)
```

## 📹 Frame Sampling Strategy
```
⚠️ VERIFIED through code inspection (not trusting FRAME_PROCESSING_PYTHON_ONLY.md)

Sampling Rate: X FPS (verified in unified_frame_manager.py:line X)
Total Frames Processed: ~X frames for 60s video
Sampling Method: Every Nth frame / Time-based / Adaptive

Actual Implementation:
# Include the actual code or pseudocode showing how frames are sampled
if service_name == 'example_service':
    max_frames = X
    # For a 60s video at 30fps (1800 frames):
    # step = 1800 / X = Y
    # Takes every Yth frame: [0, Y, 2Y, 3Y, ...]

    # Show exact frame indices that would be processed
    # Example: frames [0, 10, 20, 30, ...]

Implementation Location:
└── /rumiai_v2/processors/unified_frame_manager.py
    └── extract_frames() or get_frames_for_service() method
        └── Line numbers: X-Y
        └── Parameters: max_frames, strategy, etc.

Rationale: [Why this sampling rate - discovered through code/comments]
Trade-offs: [Speed vs accuracy - measured impact]
⚠️ Known Issues: [Any problems with current strategy]
```

## 🔍 Self-Containment Check
- [ ] Works without precompute imports (tested)
- [ ] No circular dependencies (verified)
- [ ] Clear service boundaries (documented below)
- [ ] Isolated test exists: /test_[service]_isolation.py

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                     OUTPUT
[Input Type] ─────────> [Service Process] ─────────> [Output Structure]
                           ├── Component 1
                           ├── Component 2
                           └── Component 3
```

### Data Flow Pipeline
```
1. Input Stage
   └── [Describe input processing]

2. Processing Stage
   ├── Step 1: [What happens]
   ├── Step 2: [What happens]
   └── Step 3: [What happens]

3. Output Stage
   └── {output_structure}
```

### Component Breakdown (for services with multiple outputs like MediaPipe)
#### Component 1 (e.g., Face Detection)
- **Output**: What data structure
- **Processing time**: X% of total
- **Features enabled**: List what Phase 2 features use this

#### Component 2 (e.g., Pose Estimation)
- **Output**: What data structure
- **Processing time**: X% of total
- **Features enabled**: List what Phase 2 features use this

### Timeline Integration
```python
# How this service integrates with timeline_builder.py
timeline_builder.py:add_[service]_entries() (line X)
├── Entry type: '[type_name]'
├── Data structure: {timestamp, ...}
└── Validation: [any special handling]
```

## 📁 File Structure & Key Locations
```
Service Implementation:
├── /rumiai_v2/api/[service_file].py (main implementation)
├── /rumiai_v2/api/ml_services_unified.py (lines X-Y)
└── /rumiai_v2/ml_services/[service]_service.py (if exists)

Timeline Integration:
└── /rumiai_v2/processors/timeline_builder.py
    └── add_[service]_entries() (lines X-Y)

Temporal Processing:
└── /rumiai_v2/processors/temporal_compute.py
    └── [Where this service's data is used] (lines X-Y)

Configuration:
└── /rumiai_v2/config/[service]_config.py (if exists)

Tests:
├── /test_[service]_isolation.py
└── /benchmark_[service].py
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios
| Failure | Cause | Impact | Recovery | Frequency |
|---------|-------|--------|----------|------------|
| [Service timeout] | [Processing too long] | [Partial/no results] | [Return partial or skip] | [X% of videos] |
| [Model load fail] | [Missing files/deps] | [Service unavailable] | [Download/install, retry] | [First run/rare] |
| [Memory overflow] | [Insufficient RAM] | [Process killed] | [Reduce quality/batch size] | [X% on low-mem] |
| [Input invalid] | [Corrupt/unsupported] | [Processing fails] | [Validate input, skip] | [X% of inputs] |
| [Dependency crash] | [External tool fails] | [Service fails] | [Retry or fallback] | [Rare/X%] |

### Graceful Degradation Strategy
- **Principle**: Service failures should not crash the pipeline
- **Empty Results**: Return valid but empty structure when service fails
- **Logging**: All failures logged with context for debugging
- **Pipeline Continuation**: Other services continue even if this one fails
- **Status Indication**: Output includes success/failure status

### Monitoring Recommendations
- **Key Metrics**: Success rate, timeout frequency, average processing time
- **Alerts**: Set up for >X% failure rate or >Y timeouts per hour
- **Logs**: Monitor for specific error patterns

## 🐛 Current Issues & Future Fixes

### Priority: HIGH 🔴
- **Issue**: [Specific problem]
- **Impact**: [How it affects performance/reliability]
- **Current Workaround**: [If any]
- **Proposed Fix**: [Solution]
- **Effort Estimate**: X days
- **Files Affected**: [List files]

### Priority: MEDIUM 🟡
- **Issue**: [Specific problem]
- **Impact**: [How it affects performance/reliability]
- **Proposed Fix**: [Solution]
- **Effort Estimate**: X days

### Priority: LOW 🟢
- **Issue**: [Specific problem]
- **Impact**: [Minor effect]
- **Proposed Fix**: [Solution]
- **Effort Estimate**: X days

## 🧪 Testing & Validation

### Two Types of Testing

#### 1. Functional Testing (Isolation)
- **Purpose**: Verify service works correctly
- **NOT for**: Performance measurement
- **Use**: Debugging, dependency verification, functionality checks
- **Location**: `/test_[service]_isolation.py`

#### 2. Performance Testing (Full Pipeline)
- **Purpose**: Measure actual performance with resource competition
- **Status**: Pending proper instrumentation setup
- **Metrics**: Currently marked as TBC (To Be Confirmed)
- **Note**: Individual service timing requires instrumentation

### CRITICAL: Cold Start Testing Protocol
**All performance measurements MUST use cold start conditions to reflect real-world usage.**

#### Test Videos (Standard Set)
- **18s video**: https://www.tiktok.com/@meganlock_/video/7428596413707144481
- **73s video**: https://www.tiktok.com/@janemukbangs/video/7500252920844193067
- **120s video**: https://www.tiktok.com/@jakedoesmusicsometimes/video/6923023955813092613

#### Cache Clearing Requirements (Before EACH Test)
```bash
# 1. Clear frame caches
rm -rf /tmp/rumiai_frames_*
rm -rf /tmp/tmp*frame*.jpg

# 2. Clear video downloads
rm -rf /home/jorge/rumiaifinal/temp/*.mp4

# 3. Clear test outputs
rm -rf /tmp/vision_test_*
rm -rf /home/jorge/rumiaifinal/insights/test_*

# 4. Clear Python cache
find /home/jorge/rumiaifinal -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 5. Kill all Python processes
pkill -f python3

# 6. Wait for system to settle
sleep 2
```

#### Production Test Command
```bash
# This is what we're actually measuring:
python3 scripts/rumiai_runner.py 'VIDEO_URL'
```

#### Cold Start Test Script
```bash
# Use the comprehensive cold start tester
python3 cold_start_performance_test.py

# This script:
# - Clears ALL caches before each video
# - Runs production command
# - Measures total end-to-end time
# - Tests with 18s, 73s, 120s videos
# - Provides real-world performance metrics
```

#### What NOT to Do
❌ **Do not** run multiple tests in sequence without clearing cache
❌ **Do not** use warm cache results for documentation
❌ **Do not** estimate performance - measure it
❌ **Do not** test services in isolation for performance metrics - use full pipeline

## 📈 Optimization Opportunities
- [ ] **Frame Batching**: Process multiple frames simultaneously within video
- [ ] **GPU Acceleration**: Enable CUDA/GPU processing where available
- [ ] **Caching**: Cache results for repeated frame analysis
- [ ] **Note**: Video batching (multiple videos in parallel) is NOT a current optimization goal

## 🔄 Dependencies
```
External Libraries:
├── [library1] version X.X (purpose)
├── [library2] version X.X (purpose)
└── [library3] version X.X (purpose)

Internal Dependencies:
├── UnifiedFrameManager (frame extraction)
├── [Other internal] (purpose)
└── [Other internal] (purpose)
```

---
```

## 📊 DISCOVERY COMMANDS

For each service, run these discovery commands:

### 1. Find Implementation
```bash
# Locate main service file
find /home/jorge/rumiaifinal -name "*[service_name]*.py" -type f

# Find service usage
grep -r "[service_name]" /home/jorge/rumiaifinal/rumiai_v2 --include="*.py"

# Check imports
grep -r "import.*[service_name]" /home/jorge/rumiaifinal/rumiai_v2
```

### 2. Verify Frame Sampling
```bash
# Check frame extraction
grep -n "extract_frames\|fps\|FPS" /home/jorge/rumiaifinal/rumiai_v2/processors/unified_frame_manager.py

# Find service-specific sampling
grep -n "fps.*=\|FPS.*=" [service_file].py
```

### 3. Functional Testing (Isolation)
```python
# Test service functionality (NOT for performance metrics)
import asyncio
from rumiai_v2.api.[service] import [ServiceClass]

# Verify service works correctly
async def test():
    service = [ServiceClass]()
    result = await service.process('test_video.mp4')
    assert result is not None
    print('✅ Service functional')
    print('⚠️ Note: Isolation timing is NOT production performance')

asyncio.run(test())

# Performance testing must be done via full pipeline
# with proper instrumentation (TBC)
```

### 4. Timeline Integration
```bash
# Find timeline integration
grep -n "add.*entries\|timeline" /home/jorge/rumiaifinal/rumiai_v2/processors/timeline_builder.py

# Check entry types
grep -n "entry_type.*=\|'entry_type'" /home/jorge/rumiaifinal/rumiai_v2/processors/timeline_builder.py
```

### 5. Self-Containment Test
```python
# Test without precompute (functionality verification only)
# 1. Comment out all precompute imports
# 2. Run service in isolation
# 3. Verify it still works functionally
# Note: This is NOT for performance measurement
```

## ✅ VALIDATION CHECKLIST

Before marking a service section complete:

### Discovery
- [ ] Ran all discovery commands
- [ ] Verified frame sampling rate through code
- [ ] Tested performance with 60s and 120s videos
- [ ] Confirmed self-containment (no precompute)

### Documentation
- [ ] All file paths verified and correct
- [ ] Line numbers accurate
- [ ] Performance metrics marked as TBC (awaiting instrumentation tests)
- [ ] Issues categorized by priority
- [ ] No legacy information copied without verification

### Testing
- [ ] Isolation test passes (functionality verified)
- [ ] Performance benchmarks marked TBC (pending instrumentation)
- [ ] Memory usage marked TBC (pending instrumentation)
- [ ] GPU compatibility verified

## 🤝 USER CONSULTATION POINTS

During documentation, consult user when:
1. **Performance is surprisingly bad** - Should we investigate deeper?
2. **Major architectural issues found** - Should we fix now or document for later?
3. **Service seems redundant** - Is this still needed?
4. **Legacy claims conflict with implementation** - Which is correct?
5. **Optimization opportunities found** - Priority for implementation?

## 📝 NOTES FOR PHASE 2 PREPARATION

While documenting services, note:
- Which features each service enables (for Phase 2 feature docs)
- Cross-service dependencies (for Phase 3 architecture)
- Deprecated features that no longer exist
- New features not documented in legacy

---

**REMEMBER**:
1. Verify EVERYTHING from legacy docs through code inspection
2. Run actual performance tests, don't trust claims
3. Document what IS, not what legacy docs say SHOULD BE
4. Consult user when uncertain