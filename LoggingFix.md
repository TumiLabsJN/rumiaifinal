# Stage 7 Logging & Debugging Fix

**Date**: 2025-10-24
**Status**: 🔴 ACTIVE - Logging configuration preventing diagnosis of Bug #3
**Context**: Cannot see Claude API responses due to INFO-level logging not outputting

---

## Problem Statement

**Objective**: Capture Claude's actual API response to diagnose why JSON parsing fails despite successful API calls (1000+ tokens returned).

**Current Blocker**: Added `logger.info()` statements to capture response details (lines 276-279), but they don't appear in output.

**Evidence**:
```python
# Lines 276-279 in stage7_llm_analysis.py (ADDED BUT NOT VISIBLE)
logger.info(f"{window_type}: API call completed, processing response...")
logger.info(f"{window_type}: Response type: {type(response)}")
logger.info(f"{window_type}: Response.content type: {type(response.content)}")
logger.info(f"{window_type}: Response.content length: {len(response.content)}")
```

**What IS visible**:
- ✅ WARNING-level messages (e.g., "Feature X missing distribution data")
- ✅ ERROR-level messages (e.g., "JSON parse error: Expecting value...")
- ❌ INFO-level messages (our debug logs)

---

## Root Cause Analysis

### Hypothesis 1: Logger Level Not Set (MOST LIKELY)

**Current Logger Configuration** (line 32):
```python
logger = logging.getLogger("rumiai.stage7_llm_analysis")
# No setLevel() called → defaults to WARNING
```

**Problem**: Python's default logging level is WARNING. Without explicitly setting to INFO or DEBUG, `logger.info()` messages are filtered out.

**Evidence**:
- `logger.warning()` works (we see "Feature X missing distribution data")
- `logger.error()` works (we see "JSON parse error")
- `logger.info()` doesn't work (our debug messages invisible)

---

### Hypothesis 2: No Console Handler Configured

Even if logger level is set, there might be no handler to output to console.

**Current State**: No handlers configured in stage7_llm_analysis.py

**Result**: Messages might be logged but not displayed.

---

### Hypothesis 3: Python Module Caching

Python might be using a cached `.pyc` file instead of the updated source.

**Less Likely**: We verified the changes exist in the `.py` file.

---

## Solution Options

### **Solution 1: Configure Logging at Module Level** ⭐ RECOMMENDED

**Complexity**: Low (3-5 lines)
**Impact**: Enables INFO-level logging for debugging
**Location**: `ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py`

**Implementation**:

```python
# Add after line 32 (after logger = logging.getLogger(...))

# Configure logger for debugging (can be removed after fixing Bug #3)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
```

**Why This Works**:
- Sets logger level to INFO (shows info() messages)
- Adds console handler (outputs to stdout/stderr)
- Conditional check prevents duplicate handlers
- Can be easily removed after debugging

**Test**:
```bash
python -c "
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import logger
logger.info('Test INFO message')
logger.warning('Test WARNING message')
"
# Expected: Both messages appear
```

---

### **Solution 2: Use Print Statements** (Quick & Dirty)

**Complexity**: Trivial (1 line change)
**Impact**: Immediate visibility, but unprofessional
**Location**: Lines 276-279

**Implementation**:

```python
# Replace logger.info() with print()
print(f"{window_type}: API call completed, processing response...")
print(f"{window_type}: Response type: {type(response)}")
print(f"{window_type}: Response.content type: {type(response.content)}")
print(f"{window_type}: Response.content length: {len(response.content)}")
```

**Pros**:
- ✅ Guaranteed to work (print always outputs)
- ✅ No configuration needed
- ✅ Fast to implement

**Cons**:
- ❌ Unprofessional (not using logging framework)
- ❌ Harder to filter/control output
- ❌ Must be removed before committing

**Use Case**: Emergency debugging when logging config is unclear

---

### **Solution 3: Write to Debug File**

**Complexity**: Low (5-10 lines)
**Impact**: Guaranteed capture, works regardless of logging config
**Location**: Lines 276-282

**Implementation**:

```python
# Add after line 273 (after response = client.messages.create(...))

# Debug: Write response to file for inspection
debug_file = f"/tmp/stage7_debug_{window_type}.txt"
with open(debug_file, 'w') as f:
    f.write(f"Response type: {type(response)}\n")
    f.write(f"Response dir: {dir(response)}\n")
    f.write(f"Response.content type: {type(response.content)}\n")
    f.write(f"Response.content: {response.content}\n")
    if hasattr(response, 'content') and len(response.content) > 0:
        f.write(f"Response.content[0]: {response.content[0]}\n")
        if hasattr(response.content[0], 'text'):
            f.write(f"Response.content[0].text length: {len(response.content[0].text)}\n")
            f.write(f"Response.content[0].text:\n{response.content[0].text}\n")
logger.info(f"Debug output written to {debug_file}")
```

**Why This Works**:
- ✅ Bypasses logging configuration entirely
- ✅ Guaranteed to capture response
- ✅ Persistent (can inspect after crash)
- ✅ Shows exactly what Claude returned

**Test**:
```bash
# After running Stage 7
cat /tmp/stage7_debug_hook.txt
# Shows complete response details
```

**Cleanup**:
```bash
rm /tmp/stage7_debug_*.txt
```

---

### **Solution 4: Configure Logging Globally**

**Complexity**: Low (2 lines)
**Impact**: Affects all logging in the process
**Location**: Before importing stage7_llm_analysis

**Implementation**:

```python
# Add at the very start of the script (before imports)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Then import and run
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main
main(...)
```

**Why This Works**:
- ✅ Sets logging level for entire Python process
- ✅ Simple one-liner
- ✅ Standard Python practice

**Caveat**: Only works if added BEFORE any logging.getLogger() calls

---

## Recommended Implementation Plan

### **Phase 1: Emergency Debug (Use Solution 3)**

**Why**: Guaranteed to work, provides immediate diagnosis

**Steps**:
1. Add file-writing debug code (Solution 3) to lines 274-282
2. Run Stage 7 for hook window only
3. Inspect `/tmp/stage7_debug_hook.txt`
4. Determine if Claude is returning JSON or plain text

**Expected Time**: 5 minutes (1 API call ~15s + inspection)

**Expected Result**:
- If file contains valid JSON → Bug is in parsing/encoding
- If file contains plain text → Bug is in prompt (missing distribution data confusing Claude)

---

### **Phase 2: Fix Root Cause**

**Based on Phase 1 findings**:

**Scenario A: Claude Returns Valid JSON**
- Problem: Encoding issue or hidden characters
- Fix: Add error handling for encoding, try different JSON parsers
- Code Location: Line 285 (`analysis = json.loads(response_text)`)

**Scenario B: Claude Returns Plain Text**
- Problem: Prompt confuses Claude due to missing distribution data
- Fix Option 1: Update prompt to handle missing distribution gracefully
- Fix Option 2: Add distribution data to Stage 6 output (longer-term)
- Code Location: `stage7_prompts.py` lines 60-80 (prompt builder)

---

### **Phase 3: Cleanup & Prevent Recurrence**

**After fixing Bug #3**:

1. **Remove debug code** (file writing)
2. **Configure logging properly** (Solution 1) for future debugging
3. **Add logging best practices** to development guidelines
4. **Document in Stage7Bugs.md** (mark Bug #3 as RESOLVED)

**Logging Configuration to Keep** (for future debugging):
```python
# Add to stage7_llm_analysis.py after line 32
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # DEBUG for development, INFO for production
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)  # INFO for production, DEBUG for troubleshooting
```

---

## Testing & Validation

### **Verify Logging Works**

**Test Script** (`test_logging.py`):
```python
#!/usr/bin/env python3
import logging
import sys

# Test 1: Default configuration (should NOT show INFO)
print("=== Test 1: Default Logging ===")
logger1 = logging.getLogger("test1")
logger1.info("This INFO should NOT appear")
logger1.warning("This WARNING should appear")

# Test 2: Configured logger (should show INFO)
print("\n=== Test 2: Configured Logging ===")
logger2 = logging.getLogger("test2")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger2.addHandler(handler)
logger2.setLevel(logging.INFO)
logger2.info("This INFO should appear")
logger2.warning("This WARNING should also appear")

# Test 3: Global configuration
print("\n=== Test 3: Global Configuration ===")
logging.basicConfig(level=logging.INFO, force=True)
logger3 = logging.getLogger("test3")
logger3.info("This INFO should appear (global config)")
```

**Expected Output**:
```
=== Test 1: Default Logging ===
This WARNING should appear

=== Test 2: Configured Logging ===
This INFO should appear
This WARNING should also appear

=== Test 3: Global Configuration ===
This INFO should appear (global config)
```

---

### **Verify Stage 7 Debug Output**

**After implementing Solution 3 (file writing)**:

```bash
# Run Stage 7
export ANTHROPIC_API_KEY="..."
python -c "
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main
main('data/.../bucket_18-33s', '18-33s', 'test_vitamin')
"

# Check debug file
cat /tmp/stage7_debug_hook.txt

# Expected content:
# Response type: <class 'anthropic.types.message.Message'>
# Response.content type: <class 'list'>
# Response.content[0].text length: 1234
# Response.content[0].text:
# {
#   "clusters": [...],
#   ...
# }
```

---

## Alternative Debugging Approaches

### **Option A: Use pdb (Python Debugger)**

**If all else fails**, use interactive debugging:

```python
# Add breakpoint at line 276
import pdb; pdb.set_trace()

# Then run normally
# When breakpoint hits:
# (Pdb) print(type(response))
# (Pdb) print(response.content)
# (Pdb) print(response.content[0].text)
```

**Pros**: Complete visibility into all variables
**Cons**: Interactive, can't run unattended

---

### **Option B: Use pytest with Capture**

**Create test harness**:

```python
# test_stage7_response.py
import pytest
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import analyze_window_with_retry

def test_hook_response(caplog):
    with caplog.at_level(logging.INFO):
        result = analyze_window_with_retry(
            bucket_path='data/.../bucket_18-33s',
            window_type='hook',
            bucket='18-33s',
            hashtag='test_vitamin'
        )

    # caplog.text contains all log messages
    print(caplog.text)
    assert result is not None
```

---

## Known Issues & Workarounds

### **Issue 1: Logging in Multithreading**

**Problem**: ThreadPoolExecutor might buffer logs, causing delayed/missing output

**Workaround**: Use `logging.getLogger().handlers[0].flush()` after each log

---

### **Issue 2: Python Bytecode Caching**

**Problem**: Python might use cached `.pyc` instead of updated `.py`

**Workaround**:
```bash
# Clear cache before running
find ml_pipeline/stage7_llm_analysis -name "*.pyc" -delete
find ml_pipeline/stage7_llm_analysis -name "__pycache__" -type d -exec rm -rf {} +
```

---

### **Issue 3: Logger Inheritance**

**Problem**: Child loggers inherit parent logger's level

**Workaround**: Use `logger.propagate = False` to prevent inheritance

```python
logger = logging.getLogger("rumiai.stage7_llm_analysis")
logger.propagate = False  # Don't inherit from parent
logger.setLevel(logging.INFO)
```

---

## Post-Fix Checklist

After Bug #3 is resolved:

- [ ] Remove debug file writing code (if used)
- [ ] Keep proper logging configuration (Solution 1)
- [ ] Update Stage7Bugs.md (mark Bug #3 as RESOLVED)
- [ ] Document actual root cause in Stage7Bugs.md
- [ ] Test full pipeline (all 3 buckets)
- [ ] Verify Anthropic Console shows expected API call count
- [ ] Validate output JSONs (8 files for bucket_18-33s)
- [ ] Clean up any temporary debug files (`rm /tmp/stage7_debug_*.txt`)
- [ ] Commit fix with clear commit message

---

## Related Documentation

- **Bug Report**: Stage7Bugs.md (Bug #3)
- **Test Results**: RumiGeneralTests.md (Stage 7 section)
- **Python Logging Docs**: https://docs.python.org/3/library/logging.html
- **Anthropic SDK Docs**: https://github.com/anthropics/anthropic-sdk-python

---

## Quick Reference Commands

**Clear Python cache**:
```bash
find /home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis -name "*.pyc" -delete
find /home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis -name "__pycache__" -type d -exec rm -rf {} +
```

**Test logging configuration**:
```bash
python -c "
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test')
logger.info('INFO test')
logger.warning('WARNING test')
"
```

**Check debug files** (after implementing Solution 3):
```bash
ls -lh /tmp/stage7_debug_*.txt
cat /tmp/stage7_debug_hook.txt
```

**View recent API calls**:
```
# Open Anthropic Console → Settings → Logs
# Filter by: Last 1 hour
```

---

**Last Updated**: 2025-10-24
**Next Action**: Implement Solution 3 (file writing) to capture Claude's actual response and diagnose Bug #3
