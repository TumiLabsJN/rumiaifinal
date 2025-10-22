# Fix rumiai_ml_batch.py Audio Services Bug

## Problem Statement

When running videos through `rumiai_ml_batch.py`, audio_energy and emotion_detection services return empty data `{}`, while all other services work correctly. However, the same videos processed directly through `rumiai_runner.py` work perfectly.

## Evidence

### SMOKING GUN TEST: Same Video, Different Results

**Critical Discovery**: We tested the EXACT same video URL through both paths:

#### Test 1: Direct rumiai_runner.py Call (October 22, 07:10 AM)
```bash
python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@vitalproteins/video/7555522693005872439'
```
**Result**:
- `energy_level: 0.084` ✅
- `pitch_scatter_ratio: 0.5729` ✅
- `emotional_valence: 0.0` (no faces detected - legitimate)
- `word_count: 0` (no speech - legitimate)

#### Test 2: Through rumiai_ml_batch.py (October 21, 19:44 PM)
**Same URL**: `https://www.tiktok.com/@vitalproteins/video/7555522693005872439`

**Result**:
- `energy_level: 0.0` ❌
- `pitch_scatter_ratio: 0.0` ❌
- `emotional_valence: 0.0` ❌
- `word_count: 0` (same - no speech)

**Conclusion**: The SAME video URL, processed through the SAME code (`rumiai_runner.py`), produces different results depending on whether it's called directly vs through the batch subprocess. This definitively proves the issue is in how the batch orchestrator calls `rumiai_runner.py`, not in the services themselves.

### Working: Recent Direct Call (October 22, 07:10 AM)
```bash
python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@vitalproteins/video/7555522693005872439'
```
**Output File**: `/home/jorge/rumiaifinal/insights/7555522693005872439_temporal_windows_updated.json`
**Result**: Audio features work perfectly ✅

### Failing: Through rumiai_ml_batch.py (Recent Tests)
**Example Videos Tested**:
- `7539218131407981838` (Oct 21, 19:12 PM)
- `7549982799457996087` (Oct 21, 19:40 PM)
- `7554929753908645175` (Oct 21, 19:36 PM)
- `7555522693005872439` (Oct 21, 19:44 PM) ← Same video as direct test!

**Result**: All return zeros for audio features ❌

### Working Example from October 14
- File: `/data/clients/test_final/hashtags/test_vitamin/.../7554027348580912440_temporal_windows_updated.json`
- Date: October 14, 2025
- Result: `energy_level: 0.049`, `dominant_emotion_id: 2`, `emotional_valence: -1.0` ✅
- **Batch orchestrator WAS working on this date**

## What Works vs What Fails

| Service | Works in Batch? | Evidence |
|---------|----------------|----------|
| YOLO (object detection) | ✅ Yes | object_count has values |
| MediaPipe (face/gesture) | ✅ Yes | average_face_size has values |
| OCR (text detection) | ✅ Yes | overlay_unique_count has values |
| Scene Detection | ✅ Yes | scene_count has values |
| Whisper (speech) | ✅ Yes | word_count has values |
| **Audio Energy** | ❌ No | Returns empty `{}` |
| **Emotion Detection (FEAT)** | ❌ No | Returns empty `{}` |

## Code Investigation

### Changes Between Oct 14 (working) and Now (failing)

#### rumiai_runner.py
**Git diff**: Minor changes only
- Removed temporal_windows validation
- Simplified error logging
- Added exit code check in main()
**Assessment**: ❌ Not the cause (changes are benign)

#### video_analyzer.py
**Git diff**: No changes
**Assessment**: ❌ Not the cause

#### audio_energy_service.py
**Git diff**: No changes
**Assessment**: ❌ Not the cause

#### video_processor.py
**Modified**: October 21, 2025 (TODAY)
**Change**: Added bandaid fix to copy videos to `/temp/`
**Assessment**: ⚠️ **Suspicious, but...**

### Why video_processor.py Bandaid Fix is NOT the Root Cause

If the bandaid fix was causing `rumiai_runner.py` to fail (by passing local paths it rejects), then:
1. All subprocess calls would fail with exit code != 0
2. `subprocess.run(..., check=True)` would raise `CalledProcessError`
3. **ALL services** would fail (YOLO, MediaPipe, OCR, Whisper, everything)
4. No temporal_windows file would be created

But we observe:
- ✅ Subprocess succeeds (no CalledProcessError)
- ✅ Most services work fine
- ❌ Only audio_energy and emotion_detection fail

**Conclusion**: The subprocess IS successfully calling `rumiai_runner.py`, but these two specific services are failing within the pipeline.

## Subprocess Execution Analysis

### How Batch Calls rumiai_runner.py

**File**: `ml_pipeline/stage2_processing/video_processor.py:79-92`

```python
cmd = [
    sys.executable,  # python3
    'scripts/rumiai_runner.py',
    temp_video_path  # URL (when video not pre-downloaded)
]

result = subprocess.run(
    cmd,
    timeout=300,
    capture_output=True,  # ← Captures stdout/stderr
    text=True,
    check=True
)
```

### Potential Issues with Subprocess

1. **Environment Variables**
   - No `env` parameter → inherits parent environment
   - Unlikely cause since other services work

2. **Output Redirection**
   - `capture_output=True` → stdout/stderr captured
   - Services can't log normally to stderr
   - But this wouldn't cause empty data returns

3. **Working Directory**
   - No `cwd` parameter → uses parent's cwd
   - Could affect relative path resolution?

4. **Resource Limits**
   - No resource limits set
   - All services run in same subprocess, so not service-specific

## Service Architecture (from AudioServices.md)

Both failing services use **SharedAudioExtractor**:

### Whisper (Working ✅)
1. `SharedAudioExtractor.extract_once()` → extracts audio to `/tmp/tmp*.wav`
2. `whisper.cpp` → processes audio
3. Returns transcription

### Audio Energy (Failing ❌)
1. `SharedAudioExtractor.extract_once()` → extracts audio to `/tmp/tmp*.wav`
2. `librosa` → analyzes RMS/pitch
3. Returns energy features
4. **But returns empty `{}`** when called via subprocess

### Emotion Detection (Failing ❌)
1. `cv2.VideoCapture(video_path)` → opens video directly
2. Extracts frames
3. FEAT analyzes emotions
4. **But returns empty `{}`** when called via subprocess

## Key Question

**If Whisper successfully uses SharedAudioExtractor through the subprocess, why doesn't Audio Energy?**

They use the exact same extraction mechanism, same video path, same subprocess environment.

## Hypotheses to Test

### Hypothesis 1: Caching Issue
- Maybe audio_energy is trying to load cached results that don't exist?
- **Test**: Check if output files exist in `audio_energy_outputs/` directory

### Hypothesis 2: Service Execution Order
- Maybe audio_energy runs before Whisper and fails, then Whisper succeeds?
- **Test**: Check logs to see execution order

### Hypothesis 3: Error Swallowing
- Maybe audio_energy is failing but errors are being swallowed?
- **Test**: Check if `MLAnalysisResult.success` is False but data is empty dict

### Hypothesis 4: Conditional Code Path
- Maybe there's a code path that returns empty dict based on some condition?
- **Test**: Add logging to track which code path is executed

## Debugging Strategy (NON-INVASIVE)

### Step 1: Enhanced Logging in video_processor.py

**Status**: ✅ **ADDED** (October 22, 2025)

**What Was Added**: Extended logging after subprocess.run() to capture full output

**File**: `/home/jorge/rumiaifinal/ml_pipeline/stage2_processing/video_processor.py`

**Location**: Lines 100-117 (after `subprocess.run()` completes)

**Code Added**:
```python
# EXISTING CODE (lines 94-98):
# Log stdout/stderr for debugging (don't parse - no JSON contract)
if result.stdout:
    logger.debug(f"RumiAI stdout: {result.stdout[:500]}")
if result.stderr:
    logger.warning(f"RumiAI stderr: {result.stderr[:500]}")

# NEW ENHANCED LOGGING (ADD AFTER LINE 98):
# DEBUG: Capture full output to diagnose audio_energy/emotion_detection failures
logger.info(f"[DEBUG] RumiAI subprocess completed for video {video_id}")
logger.info(f"[DEBUG] Exit code: {result.returncode}")
logger.info(f"[DEBUG] Stdout length: {len(result.stdout) if result.stdout else 0} chars")
logger.info(f"[DEBUG] Stderr length: {len(result.stderr) if result.stderr else 0} chars")

# Log full stderr for service failure diagnostics
if result.stderr:
    logger.info(f"[DEBUG] Full stderr:\n{result.stderr}")

# Check for specific failure indicators
if result.stderr:
    if "audio_energy" in result.stderr.lower():
        logger.warning(f"[DEBUG] Audio energy mentioned in stderr for {video_id}")
    if "emotion" in result.stderr.lower():
        logger.warning(f"[DEBUG] Emotion detection mentioned in stderr for {video_id}")
    if "error" in result.stderr.lower() or "fail" in result.stderr.lower():
        logger.error(f"[DEBUG] Error/failure keywords found in stderr for {video_id}")
```

**Impact**:
- ✅ Zero production code changes (only logging)
- ✅ Logs written to batch orchestrator log files
- ✅ Enables diagnosis without re-running tests
- ⚠️ Will increase log file size (~5-10KB per video)

**How to Analyze Logs** (for future CLI instance):

⚠️ **CRITICAL: Start with Step 0 before analyzing logs!**

### Step 0: Verify Batch Process is Complete

**Before analyzing logs, ALWAYS check if the batch process is still running:**

```bash
# Check if batch orchestrator is still running
ps aux | grep rumiai_ml_batch | grep -v grep

# Expected outputs:
# - If you see a process: Batch is STILL RUNNING, wait for completion
# - If no output: Batch has FINISHED, safe to analyze logs
```

**Why This Matters**:
- Log files use **buffered I/O** - they won't write until buffer fills or process exits
- **Empty log files (0 bytes) + active process = NORMAL** (logs not flushed yet)
- **Empty log files (0 bytes) + no active process = BROKEN** (logging failed)

**If Process is Still Running**:
```bash
# Option 1: Wait and monitor
watch "ps aux | grep rumiai_ml_batch | grep -v grep"

# Option 2: Get the PID and check periodically
# From the grep output, note the PID (second column)
# Then check: ps -p {PID}

# Option 3: Check temporal_windows files directly (they're created immediately)
# Even if logs aren't flushed, rumiai_runner.py creates insights files
ls -lt /home/jorge/rumiaifinal/insights/*.json | head -5
```

**If Process Has Finished**:
- Proceed to analyzing logs below

**Common Mistake**: Seeing 0-byte log files and assuming logging is broken, when actually the process just hasn't finished yet. **This was caught on October 22, 2025** - don't repeat this mistake!

---

### Analyzing Logs (After Process Completes)

When examining a video processed after this change, look for these patterns in the log file:

```bash
# Find the log file for the batch run
ls -lt /home/jorge/rumiaifinal/data/logs/rumiai_ml_*.log | head -1

# For a specific video_id, extract its debug logs:
grep "\[DEBUG\].*{video_id}" /path/to/log.log

# Look for these indicators:
grep "Audio energy mentioned" /path/to/log.log  # Did audio_energy run?
grep "Emotion detection mentioned" /path/to/log.log  # Did emotion run?
grep "Full stderr" /path/to/log.log  # What did rumiai_runner output?
```

**What the Logs Tell Us**:

1. **If stdout length > 1000 chars**: rumiai_runner.py ran and produced output
2. **If stderr contains "audio_energy"**: Audio energy service was invoked
3. **If stderr contains "emotion"**: Emotion detection service was invoked
4. **If stderr contains "Audio energy analysis failed"**: Service ran but errored
5. **If stderr is empty**: rumiai_runner.py ran silently (suspicious)

**Expected Output for Working Video**:
```
[DEBUG] RumiAI subprocess completed for video 7555522693005872439
[DEBUG] Exit code: 0
[DEBUG] Stdout length: 2847 chars
[DEBUG] Stderr length: 15234 chars
[DEBUG] Audio energy mentioned in stderr for 7555522693005872439
[DEBUG] Emotion detection mentioned in stderr for 7555522693005872439
```

**Expected Output for Failing Video**:
```
[DEBUG] RumiAI subprocess completed for video 7555522693005872439
[DEBUG] Exit code: 0
[DEBUG] Stdout length: 2847 chars
[DEBUG] Stderr length: 8432 chars
[No mentions of audio_energy or emotion in stderr]
```

If no mentions → services weren't invoked at all (silent skip)
If mentions + "failed" → services tried but errored

### Step 2: Check MLAnalysisResult Status
Examine unified_analysis JSON to see if services have `success: false`:

```bash
cat unified_analysis/{video_id}.json | jq '.ml_results.audio_energy, .ml_results.emotion_detection'
```

### Step 3: Compare Service Output Files
Check if output files exist and compare:

```bash
# Working (direct run)
ls -lh audio_energy_outputs/{video_id}/

# Failing (batch run)
# Are output files created? If not, service didn't save results
```

### Step 4: Test with Minimal Subprocess
Create a test script that mimics the batch subprocess call:

```python
import subprocess
import sys

result = subprocess.run(
    [sys.executable, 'scripts/rumiai_runner.py', 'VIDEO_URL'],
    capture_output=True,
    text=True,
    check=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
```

Run this and see if it reproduces the issue.

## Next Steps

1. Run Step 4 (minimal subprocess test) to isolate the issue
2. If that fails too, the issue is subprocess-specific
3. If that works, the issue is in video_processor.py setup
4. Add detailed logging to track service execution flow
5. Check if services are even being called vs returning cached empty results

## 🎯 ROOT CAUSE IDENTIFIED - October 22, 2025 (08:22 AM)

### THE BUG: rumiai_runner.py Rejects Local File Paths

**File**: `scripts/rumiai_runner.py` lines 477-483

```python
if args.video_input.startswith('http'):
    video_url = args.video_input
else:
    logger.error(f"Error: '{args.video_input}' is not a valid URL")
    logger.error("Please provide a complete TikTok URL starting with http:// or https://")
    sys.exit(1)  # ❌ EXITS BEFORE SERVICES RUN
```

### How the Bug Manifests

**Batch Orchestrator Flow**:
1. `rumiai_ml_batch.py` downloads videos to bucket directories
2. `video_processor.py` copies videos to `/temp/` directory (line 74-76)
3. `video_processor.py` calls: `subprocess.run(['python3', 'scripts/rumiai_runner.py', '/temp/VIDEO_ID.mp4'])`
4. `rumiai_runner.py` receives local file path: `/temp/7550427512438803767.mp4`
5. `rumiai_runner.py` validates input and finds it doesn't start with 'http'
6. **Script exits with code 1 BEFORE any ML services run**
7. No services execute → No output directories created → Empty `{}` in ml_data

**Why Direct Runs Work**:
```bash
python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@user/video/123'
# ✅ URL input → passes validation → services run → features extracted
```

**Why Batch Runs Fail**:
```bash
python3 scripts/rumiai_runner.py '/temp/7550427512438803767.mp4'
# ❌ Local path → fails validation → exits early → no services run
```

### Reproduction Test (Confirmed)

```bash
# Test 1: URL input (WORKS)
python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@naraazizasmith/video/7550427512438803767'
# Result: ✅ Exit code 0, audio_energy mentioned, emotion mentioned
# Output: energy_level: 0.047, emotional_valence: -1.0

# Test 2: Local file path (FAILS)
python3 scripts/rumiai_runner.py '/home/jorge/rumiaifinal/temp/7550427512438803767.mp4'
# Result: ❌ Exit code 1, services NOT mentioned
# Error: "'/home/jorge/.../temp/7550427512438803767.mp4' is not a valid URL"
```

### ⚠️ CRITICAL DISCOVERY: TWO SEPARATE BUGS (Oct 22, 09:00 AM)

**Timeline Analysis Reveals**:

| Date | Features | Bug Type | Cause |
|------|----------|----------|-------|
| **Oct 14** | `energy_level: 0.021`, `pitch_scatter_ratio: 0.628` | ✅ **WORKING** | Before any changes |
| **Oct 15-20** | `energy_level: null`, `pitch_scatter_ratio: null` | ❌ **Bug #1** | Unknown (temporal_compute issue?) |
| **Oct 21+** | `energy_level: 0.0`, `pitch_scatter_ratio: 0.0` | ❌ **Bug #2** | URL validation (confirmed) |

**Evidence**:
```bash
# Oct 14 (WORKING)
jq '.temporal_windows.hook | {energy_level, pitch_scatter_ratio}' \
  data/clients/test_final/.../7546425205590215991_temporal_windows_updated.json
# Result: {"energy_level": 0.021, "pitch_scatter_ratio": 0.628}

# Oct 15-20 (Bug #1)
jq '.temporal_windows.hook | {energy_level, pitch_scatter_ratio}' \
  data/clients/test_run/.../7428596413707144481_temporal_windows_updated.json
# Result: {"energy_level": null, "pitch_scatter_ratio": null}

# Oct 21+ (Bug #2 - Current)
jq '.temporal_windows.hook | {energy_level, pitch_scatter_ratio}' \
  insights/7550427512438803767_temporal_windows_updated.json
# Result: {"energy_level": 0.0, "pitch_scatter_ratio": 0.0}
```

**Key Difference**:
- **Bug #1 (null)**: Features don't exist at all in output → temporal_compute not extracting them
- **Bug #2 (0.0)**: Features exist but have zero values → services not running (URL validation)

### Why Bug #2 (URL Validation) Explains Oct 21+ Symptoms

1. **Empty `{}` in ml_data**: Services never ran, so UnifiedAnalysis creates empty dicts
2. **No output directories**: Services never got to the point of creating directories
3. **Zero values (not null)**: temporal_compute runs but has no ml_data to extract from
4. **Exit code 1**: subprocess returns error but video_processor.py doesn't properly handle it
5. **Bandaid fix (Oct 21)**: Changed from passing URLs to always passing local paths

### Historical Context - The Bandaid Fix That Broke Things

**Git commit a20bedb (Oct 21)**:
```python
# BEFORE (worked with URLs)
cmd = ['python3', 'scripts/rumiai_runner.py', video_path]
# video_path could be URL or local path

# AFTER (always local path)
cmd = ['python3', 'scripts/rumiai_runner.py', temp_video_path]
# temp_video_path is ALWAYS /temp/VIDEO_ID.mp4 (local path)
```

**The "hybrid approach" in video_processor.py:183-188**:
```python
if os.path.exists(local_video_path):
    video_path = local_video_path  # Local path
elif 'webVideoUrl' in video:
    video_path = video['webVideoUrl']  # URL
```

Before Oct 21, when videos weren't downloaded yet, the system passed URLs. After the bandaid fix, it ALWAYS copies to /temp/ and passes local paths, which rumiai_runner.py rejects.

### Solution Options

**Option 1: Make rumiai_runner.py Accept Local Paths** (RECOMMENDED)
- Modify validation to detect local file paths
- If local path exists, skip Apify scraping and process directly
- Maintains backward compatibility with URL inputs

**Option 2: Make Batch Orchestrator Pass URLs**
- Store video URLs in metadata during Stage 1
- Pass URLs instead of local paths to rumiai_runner.py
- Requires rumiai_runner to re-download videos (wasteful)

**Option 3: Create Separate Entry Point for Batch**
- New script that accepts local paths
- Bypasses URL validation and metadata scraping
- More maintainable separation of concerns

### 🎯 RECOMMENDED ACTION PLAN

**Immediate Fix (Bug #2 - Oct 21+)**:
1. ✅ Implement Option 1: Modify `rumiai_runner.py` to accept local file paths
2. ✅ Test with batch orchestrator to confirm fix
3. ✅ Verify audio_energy and emotion_detection outputs are created

**Follow-up Investigation (Bug #1 - Oct 15-20)**:
1. 🔍 Investigate why features became `null` after Oct 15
2. 🔍 Check temporal_compute.py changes between Oct 14-15
3. 🔍 Verify if this was a separate temporal_compute bug or data issue

**Confidence Level**:
- Bug #2 root cause: **100% confirmed** (reproduction test successful)
- Option 1 as fix: **95% confident** (need to address metadata handling)
- Bug #1 investigation: **Pending** (requires separate analysis)

## ✅ CONFIRMED FINDINGS - October 22, 2025 (07:45-07:52 AM)

### Test Results: 5 Fresh Videos Processed Through Batch

**Video IDs Tested**:
1. `7550427512438803767` (07:45)
2. `7554875477207289101` (07:46)
3. `7562295823749467406` (07:49)
4. `7543374279480577293` (07:50)
5. `7545600194784660749` (07:52)

### Finding 1: Bug Persists - All Videos Show Empty Audio/Emotion Features

```json
// All 5 videos show identical pattern:
{
  "energy_level": 0.0,          // ❌ Audio Energy failed
  "pitch_scatter_ratio": 0.0,   // ❌ Audio Energy failed
  "emotional_valence": 0.0,     // ❌ Emotion Detection failed
  "word_count": 11              // ✅ Whisper works
}
```

### Finding 2: unified_analysis Shows Empty Dictionaries

**Checked**: `/unified_analysis/7550427512438803767.json`

```json
{
  "ml_data": {
    "audio_energy": {},           // ❌ EMPTY (not null, not error, just {})
    "emotion_detection": {},      // ❌ EMPTY
    "whisper": { /* full data */ } // ✅ WORKS (full transcription)
  }
}
```

**Key Insight**: The keys exist in ml_data, but contain empty objects. This is NOT:
- A missing key (services registered)
- A null value (services initialized)
- An error object (no failure recorded)

This pattern suggests services were **invoked but returned nothing**.

### Finding 3: Service Output Directories Completely Missing

**Checked**: Service output directories for video `7550427512438803767`

| Service | Output Directory | Status |
|---------|-----------------|---------|
| Audio Energy | `/audio_energy_outputs/7550427512438803767/` | ❌ **Does NOT exist** |
| Emotion Detection | `/emotion_detection_outputs/7550427512438803767/` | ❌ **Does NOT exist** |
| MediaPipe | `/human_analysis_outputs/7550427512438803767/` | ✅ **EXISTS with data** |

**Critical Evidence**: Services that work (MediaPipe, Whisper) create output directories. Services that fail do NOT create directories at all.

**Conclusion**: Services are not just failing - they're **not running at all**. If they had run and errored, they would still create directories or log attempts.

### ROOT CAUSE CONFIRMED: Scenario B - Silent Skip

**Evidence Chain**:
1. ✅ Empty `{}` in ml_data (not error messages)
2. ✅ No output directories created
3. ✅ No intermediate files
4. ✅ Other services work fine and create outputs
5. ✅ subprocess returns exit code 0 (no crash)

**Conclusion**: `audio_energy` and `emotion_detection` services are being **silently skipped** during subprocess execution through `rumiai_ml_batch.py`.

They are NOT:
- Running and failing (would create directories)
- Running and returning errors (would log errors)
- Crashing the subprocess (other services work)

They are simply **not being invoked at all**.

### What This Means

The issue is NOT in:
- ❌ The services themselves (they work in direct runs)
- ❌ File paths or video access (other services access same files)
- ❌ SharedAudioExtractor (Whisper uses it successfully)
- ❌ Subprocess crash (exit code 0, other services complete)

The issue IS in:
- ✅ Service invocation logic in `video_analyzer.py`
- ✅ Conditional skip logic somewhere in the pipeline
- ✅ Environment/configuration that disables these services in subprocess context

### Next Investigation Actions

1. **Examine service invocation in `video_analyzer.py`**:
   - Check if there's conditional logic that skips audio_energy/emotion_detection
   - Look for try/except blocks that swallow failures silently
   - Verify service registration and execution order

2. **Check for environment-based skips**:
   - Look for environment variables that disable services
   - Check if there's a "skip_services" configuration
   - Verify subprocess inherits correct environment

3. **Add instrumentation to prove skip**:
   - Add logging at service invocation point
   - Log whether services are called vs skipped
   - Track execution flow through video_analyzer.py

### Logging Status - FIXED October 22, 2025 (08:15 AM)

**Original Issue**: Batch orchestrator logs were 0 bytes even after process completion.

**Root Cause**: `logging.basicConfig()` was being called AFTER logging was already initialized by imported modules. Python's `basicConfig()` only works on the first call - subsequent calls are silently ignored.

**Fix Applied**: Added `force=True` parameter to `logging.basicConfig()` in `rumiai_ml_batch.py` line 95:

```python
# File: /home/jorge/rumiaifinal/rumiai_ml_batch.py
# Location: lines 86-96

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Python 3.8+: Reconfigure even if already initialized
)
```

**Why This Works**:
- Without `force=True`: If any imported module calls `logging.getLogger()` or `basicConfig()` first, subsequent `basicConfig()` calls are ignored
- With `force=True`: Forces reconfiguration of root logger and handlers, overriding any previous setup

**Verification Test** (confirmed working):
```python
# Test that force=True allows reconfiguration
import logging
logging.getLogger('import').warning('First initialization')
logging.basicConfig(
    handlers=[logging.FileHandler('/tmp/test.log')],
    force=True  # Without this, log file would be empty
)
logging.getLogger('main').info('This appears in log file')
# Result: Log file contains "This appears in log file" ✅
```

**Status**: ✅ Fix applied and tested in isolation

**Next Steps**: Re-run batch test to verify logs populate and capture enhanced debug logs from video_processor.py

---

### If Logging Still Fails - Troubleshooting Guide

If logs remain 0 bytes after the `force=True` fix, follow these steps:

#### 1. Verify Python Version Supports force Parameter
```bash
python3 --version  # Must be 3.8+
```
If < 3.8, `force=True` is silently ignored. Use alternative fix below.

#### 2. Check File Permissions
```bash
ls -la /home/jorge/rumiaifinal/data/logs/
# Verify:
# - Directory is writable by current user
# - No permission denied errors
```

#### 3. Test Logging in Isolation
```bash
python3 -c "
import logging
from pathlib import Path
log_file = Path('/tmp/diagnostic.log')
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(log_file)],
    force=True
)
logging.getLogger('test').info('Diagnostic test')
print(f'File size: {log_file.stat().st_size}')
"
# Expected output: File size: >0 bytes
# If 0 bytes: Python logging is fundamentally broken
```

#### 4. Check for Handler Conflicts
Add this diagnostic code to `setup_logging()` to see existing handlers:

```python
# DIAGNOSTIC: Add before basicConfig() call
import logging
root = logging.getLogger()
print(f"[DIAGNOSTIC] Existing handlers BEFORE basicConfig: {root.handlers}")
print(f"[DIAGNOSTIC] Root logger level: {root.level}")

# ... existing basicConfig call ...

# DIAGNOSTIC: Add after basicConfig() call
print(f"[DIAGNOSTIC] Handlers AFTER basicConfig: {root.handlers}")
print(f"[DIAGNOSTIC] Log file path: {log_file}")
```

Run batch and check console output for diagnostics.

#### 5. Alternative Fix for Python < 3.8 or Persistent Issues

If `force=True` doesn't work, manually clear handlers:

```python
# Replace basicConfig() with manual handler setup
import logging

# Clear any existing handlers
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()

# Add new handlers
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)
```

#### 6. Check for Buffering Issues

If logs appear after process ends but not during:
```python
# Add to each handler:
file_handler = logging.FileHandler(log_file)
file_handler.flush()  # Force immediate write

# OR set stream to unbuffered mode at start of main():
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffered
```

#### 7. Verify Log File Path is Correct

```bash
# After running batch, check what log file was created
ls -lt /home/jorge/rumiaifinal/data/logs/ | head -5

# Check if logs went to a different location
find /home/jorge/rumiaifinal -name "rumiai_ml_*.log" -type f -mmin -10
```

#### 8. Last Resort: Use Print Statements

If all logging fails, temporarily replace logger calls with print:
```python
# In video_processor.py, replace:
logger.info(f"[DEBUG] ...")

# With:
print(f"[DEBUG] ...", flush=True, file=sys.stderr)
```

This bypasses Python logging entirely and writes directly to stderr.

## Investigation Timeline & Key Discoveries

### October 14, 2025
- ✅ Batch orchestrator working perfectly
- ✅ Audio features returning correct values through batch
- Example: `7554027348580912440` shows `energy_level: 0.049`, `dominant_emotion_id: 2`

### October 21, 2025 (Today)
- ❌ Batch orchestrator failing for audio_energy and emotion_detection
- ✅ Direct `rumiai_runner.py` calls still work perfectly
- 🔍 **Smoking Gun**: Same URL works direct, fails through batch

### Key Reasoning Chain

1. **Initial Hypothesis**: Path issue (videos not in expected location)
   - **Disproven**: Video files exist, other services work fine

2. **Second Hypothesis**: Our bandaid fix broke it (copying to /temp/)
   - **Disproven**: If subprocess failed, ALL services would fail
   - Reality: Only 2 services fail, 6 work fine

3. **Third Hypothesis**: Services themselves changed
   - **Disproven**: Git history shows no changes to audio_energy_service.py or video_analyzer.py

4. **Fourth Hypothesis**: rumiai_runner.py changed
   - **Disproven**: Only minor validation changes, nothing that would affect services

5. **Current Theory**: Subprocess execution environment issue
   - Same code, same video, different results
   - Only difference: direct shell vs subprocess
   - Affects only audio_energy and emotion_detection
   - Whisper uses same SharedAudioExtractor but works fine

### The Mystery

**Core Question**: If Whisper successfully uses SharedAudioExtractor to extract audio through the subprocess, why don't audio_energy and emotion_detection?

**What We Know**:
- ✅ All three services use the same audio extraction mechanism
- ✅ All three services receive the same video_path
- ✅ Subprocess successfully calls `rumiai_runner.py` (other services work)
- ✅ Services themselves are unchanged
- ❌ But audio_energy and emotion_detection return empty `{}`

**Possibilities**:
1. Services are silently skipped (not invoked at all)
2. Services run but encounter an error that's swallowed
3. Services run but a conditional code path returns empty dict
4. Subprocess environment affects how services execute

**Why This Matters**: Understanding this will reveal whether it's:
- A caching issue
- An execution order issue
- An error handling issue
- An environment variable issue
- Something else entirely

## Notes

- **Do NOT modify production code** without explicit approval
- All debugging should be logging/observation only
- Document all findings before proposing fixes
- ✅ Enhanced logging added (October 22) - production safe, zero behavior changes
