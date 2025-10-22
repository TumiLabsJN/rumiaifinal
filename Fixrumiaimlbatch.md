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

### Logging Status

**Note**: Batch orchestrator logs are empty (0 bytes) because the process is still running (PID 338550). Logs will be available after completion, but the enhanced logging added on Oct 22 should capture stderr from rumiai_runner.py subprocess calls.

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
