# Golden Dataset Testing Workflow

## Purpose
Validate that video analysis is working as intended by comparing actual outputs against expected values derived from structured video descriptions.

## Workflow Steps

### 1. Create Video Description (You)
Create a markdown file describing the video's content and expected analysis results:

```markdown
# VIDEO TEST #1

## Video Overview
- Duration: 30 seconds
- Type: Tutorial/Educational
- Subject: Person speaking to camera with text overlays

## Expected Analysis

### Scene Structure
- Scene 1 (0-5s): Introduction with face visible
- Scene 2 (5-15s): Main content with graphics
- Scene 3 (15-25s): Examples with text overlays
- Scene 4 (25-30s): Call to action

### Speech Content
- Continuous speech throughout
- Approximately 150 words
- Clear pronunciation
- No background music

### Visual Elements
- Text overlays: 10 total
- Face visible: 80% of video
- Scene changes: 3
- Dominant framing: Medium shot

### Emotional Tone
- Primary emotion: Enthusiastic/Happy
- Secondary: Informative/Neutral
- Transitions: 2 (intro excitement → calm explanation → energetic CTA)
```

### 2. Generate Expected JSON (Claude)
Based on your .md description, I will generate a comprehensive expected output JSON:

```json
{
  "creative_density": {
    "CoreMetrics": {
      "sceneChangeCount": 3,
      "totalElements": 120,  // Calculated from overlays + faces + objects
      ...
    }
  },
  ...
}
```

The JSON will include:
- **Exact values** for clearly defined metrics (scene changes, word count)
- **Ranges** for ML-dependent values (emotions, objects)
- **All 8 analysis types** with relevant metrics

### 3. Run Video Analysis (You)
```bash
# Process the test video
python3 scripts/local_video_runner.py test_videos/video_test_1.mp4
```

This creates actual analysis outputs in:
```
insights/video_test_1/
├── creative_density/
├── emotional_journey/
├── speech_analysis/
└── ... (all 8 types)
```

### 4. Compare Results (You)
```bash
# Run comparison
python3 scripts/compare_golden.py video_test_1 expected_outputs/video_test_1.json
```

### 5. Review Report (Together)
The comparison generates a detailed report showing:

```
======================================================================
🔍 GOLDEN DATASET COMPARISON REPORT
======================================================================

📊 CREATIVE_DENSITY
==================================================
CoreMetrics:
  ✅ EXACT                 sceneChangeCount    Expected: 3, Actual: 3
  ⚠️ WITHIN TOLERANCE      totalElements       Expected: 120, Actual: 115
  ❌ OUTSIDE TOLERANCE     avgDensity          Expected: 10.5, Actual: 8.2

📊 EMOTIONAL_JOURNEY
==================================================
CoreMetrics:
  ✅ EXACT                 dominantEmotion     Expected: happy, Actual: happy
  ⚠️ WITHIN TOLERANCE      emotionalIntensity  Expected: 0.75, Actual: 0.72

[... continues for all 8 analyses ...]

📈 SUMMARY STATISTICS
======================================================================
Total Comparisons: 84
  ✅ Exact Matches:     45 (53.6%)
  ⚠️ Within Tolerance:  28 (33.3%)
  ❌ Outside Tolerance:  8 ( 9.5%)
  ❌ Missing Fields:     3 ( 3.6%)

✅ VALIDATION PASSED (86.9% acceptance rate)
```

## Tolerance Levels

The comparison script uses smart tolerances:

| Tolerance | Fields | Acceptable Variance |
|-----------|--------|-------------------|
| **Exact (0%)** | sceneChangeCount, totalWords, uniqueEmotions | Must match exactly |
| **Low (5%)** | speechCoverage, videoDuration | Minor timing variations |
| **Medium (10%)** | wordsPerMinute, faceVisibilityRate | ML confidence variations |
| **High (15%)** | totalElements, overlayDensity, emotionalIntensity | High ML variance expected |

## Directory Structure

```
rumiaifinal/
├── test_videos/
│   ├── VIDEO_TEST_1.md          # Your description
│   ├── video_test_1.mp4         # Actual video file
│   └── ...
├── expected_outputs/
│   ├── video_test_1.json        # Generated from .md
│   └── ...
├── insights/
│   └── video_test_1/            # Actual analysis results
└── validation_reports/
    └── video_test_1_validation_*.md  # Comparison reports
```

## Benefits

1. **Comprehensive Testing**: Validates all 8 analysis types
2. **Flexible Validation**: Smart tolerances for ML variance
3. **Clear Documentation**: .md descriptions serve as test documentation
4. **Traceable Results**: Reports show exactly what passed/failed
5. **Library Building**: Accumulate test cases over time

## Quick Start Example

```bash
# 1. You provide VIDEO_TEST_1.md with video description
# 2. I generate expected_outputs/video_test_1.json
# 3. You run:
python3 scripts/local_video_runner.py test_videos/video_test_1.mp4

# 4. Compare:
python3 scripts/compare_golden.py video_test_1 expected_outputs/video_test_1.json

# 5. Review report together
cat validation_reports/video_test_1_validation_*.md
```

## Next Steps

1. Provide your first VIDEO TEST #1.md
2. I'll generate the expected JSON
3. We'll run through the complete validation cycle
4. Refine tolerances based on actual results