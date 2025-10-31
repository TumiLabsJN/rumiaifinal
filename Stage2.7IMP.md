# Stage 2.7 Classification: Implementation Analysis & Debugging Guide

**Date:** 2025-10-31
**Test Flow:** rollo_test5/wellnesspt2_test5
**Purpose:** Document proper analysis methodology and failure investigation

---

## CRITICAL LESSON LEARNED: How to Properly Analyze Classification Results

### What Went Wrong in Initial Analysis

**INCORRECT ANALYSIS:**
- Searched logs for "❌ Failed" pattern
- Found only 5 "Extra data" errors in early log output
- Calculated 295/300 = 98.3% success rate
- **THIS WAS COMPLETELY WRONG**

**WHY IT FAILED:**
1. ❌ Only searched beginning of log file
2. ❌ Relied on log grep without verification
3. ❌ Did not check checkpoint file (ground truth)
4. ❌ Did not count actual output files
5. ❌ Made assumptions without validation

---

## CORRECT METHODOLOGY: How to Analyze Classification Results

### Step 1: Check Checkpoint File (Ground Truth)

**Location:** `.checkpoints/classification_checkpoint.json`

```bash
# Navigate to checkpoint
cd /home/jorge/rumiaifinal/data/clients/{client}/hashtags/{hashtag}/{analysis_mode}_contrastive/.checkpoints

# Count completed and failed
cat classification_checkpoint.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Completed: {len(data[\"completed\"])}')
print(f'Failed: {len(data[\"failed\"])}')
print(f'Total: {len(data[\"completed\"]) + len(data[\"failed\"])}')
print(f'Success Rate: {len(data[\"completed\"]) / (len(data[\"completed\"]) + len(data[\"failed\"])) * 100:.1f}%')
"
```

**Example Output:**
```
Completed: 241
Failed: 59
Total: 300
Success Rate: 80.3%
```

### Step 2: Verify Against Actual Output Files

```bash
# Count validated classification files per bucket
for bucket in bucket_60-90s bucket_18-33s bucket_33-60s; do
  count=$(ls content_analysis/validated/$bucket/*.json 2>/dev/null | wc -l)
  echo "$bucket: $count files"
done

# Total count
find content_analysis/validated -name "*_content.json" | wc -l
```

**Example Output:**
```
bucket_60-90s: 89 files
bucket_18-33s: 82 files
bucket_33-60s: 70 files
Total: 241 files
```

**VERIFICATION:** File count (241) MUST match checkpoint completed count (241)

### Step 3: Compare Against Manifest

```bash
# Check how many videos should have been classified
cat selection_manifest.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for bucket, videos in data['videos_by_bucket'].items():
    top = len(videos['top_performers'])
    bottom = len(videos.get('bottom_performers', []))
    print(f'{bucket}: {top} top + {bottom} bottom = {top + bottom} total')
"
```

**Example Output:**
```
60-90s: 80 top + 20 bottom = 100 total
18-33s: 80 top + 20 bottom = 100 total
33-60s: 80 top + 20 bottom = 100 total
EXPECTED TOTAL: 300
```

### Step 4: Analyze Failure Patterns

```bash
# Extract failure types from logs
grep "❌ Failed" {log_file} | sed 's/.*Failed [0-9]*\/[0-9]*: \([0-9]*\) - \(.*\)/\2/' | sort | uniq -c | sort -rn
```

**Example Output:**
```
44 Extra data: line XX column 1 (char XXX)
15 Expecting value: line 1 column 1 (char 0)
```

---

## ACTUAL RESULTS: rollo_test5/wellnesspt2_test5

### Classification Summary

| Metric | Value |
|--------|-------|
| Total Videos | 300 (3 buckets × 100 videos) |
| Successfully Classified | 241 |
| Failed | 59 |
| **Success Rate** | **80.3%** |

### Failure Breakdown by Type

| Error Type | Count | % of Failures | Root Cause |
|------------|-------|---------------|------------|
| "Extra data: line XX column 1" | 44 | 74.6% | Claude returned multiple JSON objects |
| "Expecting value: line 1 column 1" | 15 | 25.4% | Claude returned empty/non-JSON response |

### Failure Breakdown by Bucket

| Bucket | Successful | Failed | Success Rate |
|--------|------------|--------|--------------|
| 60-90s | 89 | 11 | 89.0% |
| 18-33s | 82 | 18 | 82.0% |
| 33-60s | 70 | 30 | 70.0% |

**OBSERVATION:** Failure rate increases with bucket (33-60s has 30% failure rate)

---

## FIX IMPLEMENTED: "Extra data" Errors

### Problem
Claude Haiku returned responses with **multiple JSON objects**:
```json
{
  "content_category": "food_recommendation_list",
  "hook_strategy": "numbered_list_promise"
}
{
  "extra": "object"
}
```

### Original Buggy Code
```python
# classification.py line 68-69 (OLD)
first_brace = text.find('{')       # Find FIRST {
last_brace = text.rfind('}')       # Find LAST } ← BUG!

# This extracted BOTH JSON objects, causing parse failure
json_text = text[first_brace:last_brace + 1]
```

### Fixed Code (Brace Counting)
```python
# classification.py line 79-91 (NEW)
brace_count = 0
last_brace = -1

for i in range(first_brace, len(text)):
    if text[i] == '{':
        brace_count += 1
    elif text[i] == '}':
        brace_count -= 1
        if brace_count == 0:
            # Found matching closing brace for FIRST opening brace
            last_brace = i
            break

json_text = text[first_brace:last_brace + 1]  # Extract only first JSON
```

### Impact
- **Before:** 44 "Extra data" failures (14.7% of total)
- **After (estimated):** 0 "Extra data" failures
- **Projected Success Rate:** 95% (285/300)

### Files Modified
1. `ml_pipeline/stage2_content_analysis/classification.py:67-100` (extract_json function)
2. `ml_pipeline/stage2_content_analysis/classification.py:1466` (helper function)
3. `ml_pipeline/stage2_content_analysis/classification.py:1571` (classify_caption_only)

---

## REMAINING ISSUE: "Expecting value" Errors (15 videos)

### Error Pattern
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

This means `json.loads()` received an **empty string** or **non-JSON content**.

### Investigation Plan

#### Phase 1: Identify Failed Videos

```bash
# Extract list of failed video IDs
cat .checkpoints/classification_checkpoint.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Failed video IDs:')
for vid in data['failed']:
    print(vid)
" > failed_videos.txt

# Get first 5 for analysis
head -5 failed_videos.txt > sample_failed.txt
```

#### Phase 2: Check Transcript Availability

For each failed video, check:

```bash
# Loop through sample failed videos
while read vid; do
  echo "=== Video: $vid ==="

  # Check transcript exists
  if [ -f "/home/jorge/rumiaifinal/speech_transcriptions/${vid}_whisper.json" ]; then
    echo "✓ Transcript exists"
    # Check transcript length
    wc -c "/home/jorge/rumiaifinal/speech_transcriptions/${vid}_whisper.json"
    # Check if valid
    python3 -c "import json; json.load(open('/home/jorge/rumiaifinal/speech_transcriptions/${vid}_whisper.json'))" && echo "✓ Valid JSON" || echo "✗ Invalid JSON"
  else
    echo "✗ Transcript missing"
  fi

  # Check caption exists
  if [ -f "/home/jorge/rumiaifinal/video_captions/${vid}_caption.json" ]; then
    echo "✓ Caption exists"
  else
    echo "✗ Caption missing"
  fi

  # Check validation cache status
  grep -q "\"$vid\".*\"is_valid\": true" .../transcript_validation_cache.json && echo "✓ Valid transcript" || echo "✗ Invalid transcript"

  echo ""
done < sample_failed.txt
```

#### Phase 3: Analyze API Response Patterns

Check the logs for these specific videos:

```bash
# Search for API request/response for failed videos
while read vid; do
  echo "=== Checking logs for $vid ==="

  # Find the classification attempt
  grep -A 10 -B 5 "$vid" {log_file} | grep -E "HTTP|Failed|response|timeout|error"

  echo ""
done < sample_failed.txt
```

#### Phase 4: Common Failure Patterns to Check

1. **API Timeout:**
```bash
grep -i "timeout" {log_file} | grep -f sample_failed.txt
```

2. **Rate Limiting:**
```bash
grep -i "rate.*limit\|429" {log_file} | grep -f sample_failed.txt
```

3. **Content Policy Rejection:**
```bash
grep -i "content.*policy\|inappropriate\|blocked" {log_file} | grep -f sample_failed.txt
```

4. **Empty Responses:**
```bash
grep "Raw API response.*0 chars" {log_file} | grep -f sample_failed.txt
```

#### Phase 5: Transcript Characteristics

Analyze if failed videos share common traits:

```python
import json
from pathlib import Path

# Load failed video IDs
with open('failed_videos.txt') as f:
    failed_ids = [line.strip() for line in f if 'Expecting value' in line]  # Filter for this error type

# Analyze transcripts
stats = []
for vid in failed_ids[:15]:  # First 15 "Expecting value" failures
    transcript_path = f"/home/jorge/rumiaifinal/speech_transcriptions/{vid}_whisper.json"

    if Path(transcript_path).exists():
        with open(transcript_path) as f:
            data = json.load(f)
            text = data.get('text', '')
            stats.append({
                'video_id': vid,
                'char_count': len(text),
                'word_count': len(text.split()),
                'has_music_markers': '[Music]' in text or '[music]' in text
            })
    else:
        stats.append({
            'video_id': vid,
            'char_count': 0,
            'word_count': 0,
            'has_music_markers': False,
            'note': 'NO TRANSCRIPT FILE'
        })

# Print analysis
import pandas as pd
df = pd.DataFrame(stats)
print(df.describe())
print("\nVideos with music markers:", df['has_music_markers'].sum())
```

#### Phase 6: Check Validation Cache

```bash
# Check if failed videos are marked as invalid transcripts
cat .../transcript_validation_cache.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
failed_ids = []  # Load from failed_videos.txt

for vid in failed_ids:
    if vid in data['validation_results']:
        result = data['validation_results'][vid]
        if not result['is_valid']:
            print(f'{vid}: INVALID - {result[\"failure_reason\"]}')
    else:
        print(f'{vid}: NOT IN CACHE')
"
```

### Hypothesis Testing

**Hypothesis 1: Failed videos have no valid transcripts**
- Check validation cache for all 15 failed videos
- If all are marked invalid → They should use Flow 2 (caption-only)
- Bug: Flow routing logic may be broken

**Hypothesis 2: API returned empty responses due to rate limiting**
- Check HTTP status codes in logs
- Check timing between failed requests
- Look for 429 or 5xx errors

**Hypothesis 3: Transcripts contain content that triggers Claude refusal**
- Search for controversial keywords
- Check for extremely long transcripts (>100K chars)
- Look for repeated content/spam patterns

**Hypothesis 4: Malformed API requests**
- Check prompt construction for these videos
- Verify taxonomy was properly loaded
- Check if caption/hashtag data was available

---

## NEXT STEPS FOR CONTINUATION

### Immediate Action Required

1. **Re-run classification with fixed code**
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

# Delete classification checkpoint to restart Stage 2.7
rm data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/.checkpoints/classification_checkpoint.json

# Re-run pipeline
python rumiai_ml_batch.py --client rollo_test5 --target wellnesspt2_test5 --analysis-type hashtag --analysis-mode top --selection-strategy contrastive
```

2. **Verify fix effectiveness**
```bash
# After completion, check new results
cat .checkpoints/classification_checkpoint.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'NEW Results:')
print(f'Completed: {len(data[\"completed\"])}')
print(f'Failed: {len(data[\"failed\"])}')
print(f'Success Rate: {len(data[\"completed\"]) / (len(data[\"completed\"]) + len(data[\"failed\"])) * 100:.1f}%')
"

# Count "Extra data" errors (should be 0)
grep "Extra data" {new_log_file} | wc -l

# Count "Expecting value" errors (should still be ~15)
grep "Expecting value" {new_log_file} | wc -l
```

3. **Investigate remaining "Expecting value" failures**
- Follow Phase 1-6 investigation plan above
- Document findings in `Stage2.7_ExpectingValueAnalysis.md`

### Files to Monitor

```
# Classification outputs
data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/
├── content_analysis/validated/
│   ├── bucket_60-90s/*.json
│   ├── bucket_18-33s/*.json
│   └── bucket_33-60s/*.json
├── .checkpoints/classification_checkpoint.json
└── .content_analysis_state.json

# Logs
data/logs/rumiai_ml_rollo_test5_wellnesspt2_test5_{timestamp}.log
```

### Success Criteria

- ✅ Total classified: ≥285/300 (95%)
- ✅ "Extra data" errors: 0
- ✅ "Expecting value" errors: ≤15
- ✅ All buckets: >90% success rate

---

## REFERENCE: Key Findings

### What We Learned

1. **Always check checkpoint file first** - It's the source of truth
2. **Verify with actual file counts** - Logs can be incomplete
3. **Search entire log file** - Don't assume early patterns represent all failures
4. **Count multiple error types** - Different failures need different fixes
5. **Test fixes properly** - Re-run classification to verify effectiveness

### Code Quality Issues Found

1. **Inconsistent JSON parsing** - Three different locations used `json.loads()` directly instead of centralized `extract_json()`
2. **No retry logic for empty responses** - "Expecting value" errors should trigger retry with different prompt
3. **No response validation** - Should check `response.content[0].text` is non-empty before parsing

### Recommended Future Improvements

1. Add response validation before parsing:
```python
response_text = response.content[0].text
if not response_text or not response_text.strip():
    raise ValueError("Empty API response")
```

2. Add retry logic for empty responses:
```python
if attempt < 2 and "Expecting value" in str(e):
    logger.warning(f"Empty response for {video_id}, retry {attempt+1}")
    time.sleep(2)
    continue
```

3. Add extraction statistics logging:
```python
logger.info(
    f"Extraction stats: {cleaned}/{total} responses needed cleaning "
    f"({cleaned/total*100:.1f}%)"
)
```

---

**END OF DOCUMENT**

**Note to next agent:** Follow the "CORRECT METHODOLOGY" section exactly. Do not rely on log searching alone. Always verify with checkpoint file and actual output counts.
